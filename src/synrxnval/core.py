from __future__ import annotations

import os
import re
import torch
import pandas as pd
import warnings
from rdkit import Chem
from typing import Callable

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

# ------------------------------
# TOKENIZATION
# ------------------------------
def smiles_tokens(smiles: str) -> str:
    """
    Tokenize SMILES using a regex pattern (Chemformer-style).
    Raises AssertionError if tokenization alters SMILES.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        warnings.warn(f"Invalid SMILES type or empty: {repr(smiles)}, RunTimeWarning")
        return None
    
    pattern = (
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\\|\/|:|~|@|\?|>|\*|\!|\$|\%[0-9]{2}|[0-9])"
    )
    regex = re.compile(pattern)
    tokens = regex.findall(smiles)
    tokenized_smiles = "".join(tokens)

    if smiles.replace(" ", "") != tokenized_smiles.replace(" ", ""):
        warnings.warn(
            f"Non-matching tokenization, possible exotic SMILES: {smiles}",
            RuntimeWarning,
        )

    return " ".join(tokens) if tokens else None


# ------------------------------
# OpenNMT-py interface
# ------------------------------
from onmt.bin import translate as trsl
from onmt.translate.translator import build_translator

def build_onmt_translator(
    model_path: str,
    beam_size: int = 3,
    max_length: int = 1000,
    gpu: int = 0,
    replace_unk: bool = True,
    log_probs: bool = True,
):
    """Create an OpenNMT-py Translator from a model checkpoint."""
    parser = trsl._get_parser()
    import tempfile

    with tempfile.NamedTemporaryFile("w", delete=False) as tmp_src:
        tmp_src.write("DUMMY\n")
        tmp_src_path = tmp_src.name

    opt = parser.parse_known_args([
        "--model", model_path,
        "--src", tmp_src_path,
        "--output", "/tmp/onmt_dummy_output.txt"
    ])[0]

    opt.models = [model_path]
    opt.beam_size = beam_size
    opt.n_best = beam_size
    opt.max_length = max_length
    opt.replace_unk = replace_unk
    opt.log_probs = log_probs

    if torch.cuda.is_available() and gpu >= 0:
        print("[Device] Using GPU...")
        opt.gpu = gpu
        opt.gpu_ranks = [gpu]
    else:
        print("[Device] Using CPU...")
        opt.gpu = -1
        opt.gpu_ranks = []

    translator = build_translator(opt, report_score=False)
    return translator


def inference_on_list_in_memory(
    translator,
    smi_list: list[str],
    beam_size: int = 3,
    batch_size: int = 64,
):
    """
    Perform beam search inference entirely in memory, chunked.
    Automatically splits the input list into chunks of size <= batch_size.

    Parameters
    ----------
    translator : OpenNMT-py Translator
        Pre-built translator object.
    smi_list : list[str]
        SMILES strings to translate.
    beam_size : int, default=3
        Number of hypotheses per input.
    batch_size : int, default=64
        Maximum number of reactions processed per batch.

    Returns
    -------
    predictions : list[list[str]]
        Predictions for each beam, shape [beam_size][N].
    probs : list[list[float]]
        Corresponding probabilities, same shape as predictions.
    """

    if not isinstance(smi_list, list) or not all(isinstance(s, str) for s in smi_list):
        raise ValueError("Input must be a list of SMILES strings")
    
    def chunker(seq, size):
        for pos in range(0, len(seq), size):
            yield seq[pos:pos + size]
    
    predictions = [[] for _ in range(beam_size)]
    probs = [[] for _ in range(beam_size)]

    for chunk in chunker(smi_list, batch_size):
        all_scores, all_predictions = translator.translate(
            src=chunk,
            tgt=None,
            src_dir=None,
            batch_size=batch_size,
            attn_debug=False,
        )

        for i, preds in enumerate(all_predictions):
            scores = all_scores[i]
            for b in range(min(beam_size, len(preds))):
                score = scores[b]
                prob = (
                    float(torch.exp(score.clone().detach()).item())
                    if isinstance(score, torch.Tensor)
                    else float(torch.exp(torch.tensor(score)))
                )
                predictions[b].append(preds[b])
                probs[b].append(prob)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return predictions, probs


# ------------------------------
# RDKit utilities
# ------------------------------
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

def canonicalize_smiles(smi: str) -> str:
    """Return canonical RDKit SMILES or empty string if invalid."""
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else ""


# ------------------------------
# Main class
# ------------------------------
class SynRxnVal:
    """
    Evaluate synthetic reaction plausibility using two OpenNMT-py models:
    T2 (reagent prediction) and T3 (forward prediction).
    """

    def __init__(
        self,
        input_dataset_path: str = None,
        T2_path: str = None,
        T3_path: str = None,
        beam_size: int = 3,
    ):
        self.input_dataset_path = input_dataset_path
        self.beam_size = beam_size
        self.reag_pred_translator = build_onmt_translator(model_path=T2_path)
        self.prod_pred_translator = build_onmt_translator(model_path=T3_path)

    # ---------- Data Loading and Cleaning ----------
    def load_data_into_df(self, smi_rxn_tuple_list=None, file_path=None) -> pd.DataFrame:
        """Load reactions either from tuple list or parquet file."""
        if smi_rxn_tuple_list is not None:
            pairs = [(s, r) for s, r in smi_rxn_tuple_list if isinstance(s, str) and isinstance(r, str) and s.strip()]
            return pd.DataFrame(pairs, columns=["smiles", "tag_rxn"])
        path = file_path or self.input_dataset_path
        if not path or not os.path.exists(path):
            raise ValueError("No valid input path provided.")
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        raise ValueError("Unsupported input format. Use parquet or tuples.")

    @staticmethod
    def remove_tag(smi: str) -> str:
        return smi.replace("!", "").replace("  ", " ") if isinstance(smi, str) else smi

    def remove_tag_df(self, df: pd.DataFrame, col="tag_rxn") -> pd.DataFrame:
        df[col.replace("tag_", "")] = df[col].apply(self.remove_tag)
        return df

    # ---------- Tokenization ----------
    def tokenize_col_df(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        df[f"tok_{col}"] = df[col].apply(smiles_tokens)
        return df[df[f"tok_{col}"].notnull()].reset_index(drop=True)

    # ---------- Reaction String Handling ----------
    @staticmethod
    def left_hand_side(rxn_smi: str) -> str:
        return rxn_smi.split(">")[0].strip()

    def add_left_column(self, df: pd.DataFrame, col="rxn") -> pd.DataFrame:
        df[col.replace("rxn", "reac")] = df[col].apply(self.left_hand_side)
        return df

    # ---------- Model Inference ----------
    def reag_pred_col(self, df: pd.DataFrame, col: str):
        """Run reagent prediction model on tokenized reactions."""
        smi_list = df[col].tolist()
        reags, probs = inference_on_list_in_memory(
            translator=self.reag_pred_translator,
            smi_list=smi_list,
            beam_size=self.beam_size,
        )
        return reags, probs

    @staticmethod
    def clean_reagent_beams(reags: list[list[str]]) -> list[list[str]]:
        """Clean empty reagent predictions."""
        return [[r.replace('"', "").strip() for r in beam] for beam in reags]

    @staticmethod
    def format_inputs_for_forward(left_list: list[str], reags: list[list[str]]):
        """Combine left sides and reagent predictions into forward model inputs."""
        beam_size = len(reags)
        return [
            [f"{left_list[i]} > {reags[b][i]}".strip() for i in range(len(left_list))]
            for b in range(beam_size)
        ]

    def forward_predict(self, forward_inputs: list[list[str]]):
        """Run forward model (T3*) on each reagent beam."""
        fwd_preds, fwd_probs = [], []
        for beam_inputs in forward_inputs:
            preds, probs = inference_on_list_in_memory(
                translator=self.prod_pred_translator,
                smi_list=beam_inputs,
                beam_size=self.beam_size,
            )
            fwd_preds.append(preds)
            fwd_probs.append(probs)
        return fwd_preds, fwd_probs

    # ---------- Post-processing ----------
    def select_best_prediction(
        self,
        gt_smi_list: list[str],
        forward_preds: list[list[list[str]]],
        forward_probs: list[list[list[float]]],
        forward_inputs: list[list[str]],
        canonicalize_fn: Callable[[str], str],
    ):
        """Select best forward prediction per sample based on match and probability."""
        beam_T2, beam_T3, N = len(forward_preds), len(forward_preds[0]), len(gt_smi_list)
        best_inputs, best_probs, best_preds, is_match = [], [], [], []

        for i in range(N):
            truth = canonicalize_fn(gt_smi_list[i])
            candidates, probs, inputs = [], [], []
            for b2 in range(beam_T2):
                for b3 in range(min(beam_T3, len(forward_preds[b2]))):
                    candidates.append(forward_preds[b2][b3][i])
                    probs.append(forward_probs[b2][b3][i])
                    inputs.append(forward_inputs[b2][i])

            canon_preds = [canonicalize_fn(p.replace(" ", "")) for p in candidates]
            matches = [(inp, p, pr) for inp, p, pr, cp in zip(inputs, candidates, probs, canon_preds) if cp == truth]

            if matches:
                best_inp, best_pred, best_prob = max(matches, key=lambda x: x[2])
                match_flag = True
            else:
                best_idx = max(range(len(probs)), key=lambda j: probs[j])
                best_inp, best_pred, best_prob = inputs[best_idx], candidates[best_idx], probs[best_idx]
                match_flag = False

            best_inputs.append(best_inp)
            best_probs.append(best_prob)
            best_preds.append(best_pred)
            is_match.append(match_flag)

        return best_inputs, best_probs, best_preds, is_match

    # ---------- Utility ----------
    @staticmethod
    def _combine_reaction_parts(inputs: list[str], preds: list[str]) -> list[str]:
        """Combine input 'A>B' with predicted product to 'A>B>product'."""
        return [f"{inputs[i]}>{preds[i]}".replace(" ", "") for i in range(len(inputs))]

    def add_best_pred_to_df(
        self,
        df: pd.DataFrame,
        best_inputs: list[str],
        best_probs: list[float],
        best_preds: list[str],
        is_match: list[bool],
    ) -> pd.DataFrame:
        """Attach best prediction results to dataframe."""
        df["final_rxn"] = self._combine_reaction_parts(best_inputs, best_preds)
        df["is_match"] = is_match
        df["fwd_prob"] = best_probs
        return df

    def save_reactions(self, df: pd.DataFrame, out_dir: str, base_name: str = "val_full_synthetic_reactions"):
        """
        Save the dataframe containing all information about validated reactions to Parquet

        Parameters
        ----------
        df: pd.DataFrame
        out_dir: str
            Directory to write into. Created if it does not exist.
        base_name: str, default='val_full_synthetic_reactions'
            Base filename without extension
        
        Returns
        -------
        None
        """
        os.makedirs(out_dir, exist_ok=True)
        parquet_path = os.path.join(out_dir, f"{base_name}.parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"[Done] Reactions saved to {parquet_path}")


    
    def main(self, out_dir: str = './outputs', smi_rxn_tuple_list: list[tuple[str]] | None = None, chunk_id: int | None = None):
        """
        Execute the full SynRxnVal pipeline.
        """
        print("Load and prepare data for T2...")
        df = self.load_data_into_df(smi_rxn_tuple_list=smi_rxn_tuple_list)
        df = self.tokenize_col_df(df, col='tag_rxn')
        df = self.remove_tag_df(df, col='tok_tag_rxn')
        
        print(f"Loaded df, T2 inference on {len(df)} reactions...")
        reags, _ = self.reag_pred_col(df, col='tok_rxn')
        reags = self.clean_reagent_beams(reags=reags)

        df = self.add_left_column(df, col="tok_tag_rxn")
        forward_inputs = self.format_inputs_for_forward(left_list=df["tok_tag_reac"].tolist(), reags=reags)

        print(f"T3* inference...")
        fwd_preds, fwd_probs = self.forward_predict(forward_inputs=forward_inputs)
        
        print(f"Selecting the best prediction for each input SMILES out of the {self.beam_size} x {self.beam_size} candidates...")
        best_inputs, best_probs, best_preds, is_match = self.select_best_prediction(
            gt_smi_list=df["smiles"].tolist(),
            forward_preds=fwd_preds,
            forward_probs=fwd_probs,
            forward_inputs=forward_inputs,
            canonicalize_fn=canonicalize_smiles,
        )

        df = self.add_best_pred_to_df(
            df, 
            best_inputs=best_inputs, 
            best_probs=best_probs,
            best_preds=best_preds,
            is_match=is_match,
            )
        df.drop(columns=['tag_rxn', 'tok_tag_rxn', 'tok_rxn', 'tok_tag_reac'], inplace=True)

        base_name = "val_full_synthetic_reactions"
        if chunk_id is not None:
            base_name += f"chunk{chunk_id}"

        print(f"Saving results...")
        self.save_reactions(df, out_dir, base_name=base_name)