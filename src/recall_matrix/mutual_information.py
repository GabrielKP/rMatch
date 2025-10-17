import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from recall_matrix import CONFIG, console, get_logger

log = get_logger(__name__)


class MIRM:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        """Initialize the Mutual Information Recall Matrix (MIRM) object.

        Parameters
        ----------
        model_name: str | None, default=None
            The name of the model to use.
            If None, the model will be taken from .env MODEL_NAME.
            If not specified, gpt2 will be used.
        device: str
            The device to use.
            If None, the device will be taken from .env DEVICE.
            If None, and not specifief, the device will be "auto".
            For what "auto" does, see:
            https://huggingface.co/docs/accelerate/usage_guides/big_modeling
            https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel
        """

        if model_name is None:
            model_name = CONFIG.get("MODEL_NAME", "gpt2")
        self.model_name = model_name
        log.info(f"Initializing model: {model_name}")

        if device is None:
            device = CONFIG.get("DEVICE", "auto")

        token = CONFIG.get("HF_TOKEN")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, token=token, device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=token, device_map=device
        )

        # get device
        self.device: torch.device = (
            next(iter(self.model.hf_device_map.values()))  # type: ignore
            if hasattr(self.model, "hf_device_map")
            else self.model.device
        )
        log.info(f"Using device: {self.device}")

    def cross_entropy_x_given_y(
        self,
        input_str_x: str,
        input_str_y: str,
        print_str: str,
        verbose: bool = False,
    ) -> float:
        """Compute cross_entropy(LLM(X | Y), X)"""

        # tokenize
        if len(input_str_y) > 0:
            input_str = f"{input_str_y} {input_str_x}"
            idx_char_x = len(input_str_y) + 1
        else:
            input_str = input_str_x
            idx_char_x = 0

        batch_encoding = self.tokenizer(
            input_str, return_tensors="pt", return_offsets_mapping=True
        )
        input_ids: torch.Tensor = batch_encoding.input_ids
        offset_mapping: torch.Tensor = batch_encoding.offset_mapping

        idx_id_x = min(np.where(offset_mapping[:, :, -1] > idx_char_x)[1])

        # gpt2 does not add bos tokens, and if input_str_y is
        # empty, you cannot predict the first token from nothing,
        # thus we need to start from the second token.
        if not self.tokenizer.add_bos_token and idx_id_x == 0:
            idx_id_x = 1

        if verbose:
            prefix = f"[italic red]{print_str}:[/italic red]"
            decode_str_y = self.tokenizer.decode(input_ids[0, :idx_id_x])
            decode_str_x = self.tokenizer.decode(input_ids[0, idx_id_x:])

            console.print(
                f"{prefix} [white]{decode_str_y}[/white][green]{decode_str_x}[/green]"
            )

        with torch.no_grad():
            (logits, _) = self.model(
                input_ids=input_ids.to(self.device), return_dict=False
            )

        logits = logits.cpu()
        # choose only the logits from x

        # input_ids: [... ,      y_j,      x_0,      x_1, ...,    x_n-1,      x_n]
        # labels_x :                        ^                                   ^
        labels_x = input_ids[:, idx_id_x:]
        # logits   : [... , pred_x_0, pred_x_1, pred_x_2, ..., pred_x_n, pred_x_n+1]
        # logits_x :             ^                                    ^
        logits_x = logits[:, idx_id_x - 1 : -1, :].cpu()

        logits_x_flat = logits_x.view(-1, logits_x.shape[-1])
        labels_x_flat = labels_x.view(-1)

        cross_entropy = F.cross_entropy(logits_x_flat, labels_x_flat, reduction="none")
        return cross_entropy.numpy().sum()

    def recall_matrix_rj_given_ei(
        self,
        normalize: bool,
        story_segments: list[str],
        recall_segments: list[str],
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute the recall matrix"""

        # TODO: Can do some serious optimization: batching, caching, ..
        recall_matrix = np.empty((len(story_segments), len(recall_segments)))
        for i, E_i in enumerate(tqdm(story_segments, desc="(RM)")):
            for j, R_j in enumerate(recall_segments):
                H_R_j_given_E_i = self.cross_entropy_x_given_y(
                    input_str_x=R_j,
                    input_str_y=E_i,
                    print_str="H(R_j | E_i)",
                    verbose=verbose,
                )

                H_R_j = self.cross_entropy_x_given_y(
                    input_str_x=R_j,
                    input_str_y="",
                    print_str="H(R_j)",
                    verbose=verbose,
                )

                if normalize:
                    recall_matrix[i, j] = max(0, 1 - (H_R_j_given_E_i / H_R_j))
                else:
                    recall_matrix[i, j] = max(0, (H_R_j - H_R_j_given_E_i))
                if verbose:
                    console.print(
                        (f"> MI(E_{i}, R_{j}) = {recall_matrix[i, j]}"),
                        style="bold yellow",
                    )

        return recall_matrix

    def recall_matrix_ei_given_rj(
        self,
        normalize: bool,
        story_segments: list[str],
        recall_segments: list[str],
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute the recall matrix"""

        # TODO: Can do some serious optimization: batching, caching, ..
        recall_matrix = np.empty((len(story_segments), len(recall_segments)))
        for i, E_i in enumerate(tqdm(story_segments, desc="(RM)")):
            for j, R_j in enumerate(recall_segments):
                H_E_i_given_R_j = self.cross_entropy_x_given_y(
                    input_str_x=E_i,
                    input_str_y=R_j,
                    print_str="H(E_i | R_j)",
                    verbose=verbose,
                )

                H_E_i = self.cross_entropy_x_given_y(
                    input_str_x=E_i,
                    input_str_y="",
                    print_str="H(E_i)",
                    verbose=verbose,
                )

                if normalize:
                    recall_matrix[i, j] = max(0, 1 - (H_E_i_given_R_j / H_E_i))
                else:
                    recall_matrix[i, j] = max(0, (H_E_i - H_E_i_given_R_j))
                if verbose:
                    console.print(
                        (f"> MI(E_{i}, R_{j}) = {recall_matrix[i, j]}"),
                        style="bold yellow",
                    )

        return recall_matrix

    def recall_matrix_ei_given_r0_to_rj(
        self,
        normalize: bool,
        story_segments: list[str],
        recall_segments: list[str],
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute the normalized mutual information"""

        # TODO: Can do some serious optimization: batching, caching, ..
        recall_matrix = np.empty((len(story_segments), len(recall_segments)))
        for i, E_i in enumerate(tqdm(story_segments, desc="(RM)")):
            for j, R_j in enumerate(recall_segments):
                # for printing
                if j == 0:
                    given_str = ""
                elif j == 1:
                    given_str = " | R_0"
                else:
                    given_str = f" | R_0 to R_{j - 1}"

                # compute the cross entropy
                H_E_i_given_r0_to_R_j_minus_1 = self.cross_entropy_x_given_y(
                    input_str_x=E_i,
                    input_str_y=" ".join(recall_segments[:j]),
                    print_str=f"H(E_{i}{given_str}",
                    verbose=verbose,
                )

                H_E_i_given_R_0_to_R_j = self.cross_entropy_x_given_y(
                    input_str_x=E_i,
                    input_str_y=" ".join(recall_segments[: j + 1]),
                    print_str=f"H(E_{i}{given_str}",
                    verbose=verbose,
                )

                if normalize:
                    recall_matrix[i, j] = max(
                        0, 1 - (H_E_i_given_R_0_to_R_j / H_E_i_given_r0_to_R_j_minus_1)
                    )
                else:
                    recall_matrix[i, j] = max(
                        0, (H_E_i_given_r0_to_R_j_minus_1 - H_E_i_given_R_0_to_R_j)
                    )
                if verbose:
                    console.print(
                        (f"> MI(E_{i}, R_{j}{given_str}) = {recall_matrix[i, j]}"),
                        style="bold yellow",
                    )

        return recall_matrix

    def compute_mutual_information_recall_matrix(
        self,
        mutual_information_method: str,
        normalize: bool,
        story_segments: list[str],
        recall_segments: list[str],
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute the mutual information recall matrix"""

        if mutual_information_method == "rj_given_ei":
            return self.recall_matrix_rj_given_ei(
                normalize, story_segments, recall_segments, verbose
            )
        elif mutual_information_method == "ei_given_rj":
            return self.recall_matrix_ei_given_rj(
                normalize, story_segments, recall_segments, verbose
            )
        elif mutual_information_method == "ei_given_r0_to_rj":
            return self.recall_matrix_ei_given_r0_to_rj(
                normalize, story_segments, recall_segments, verbose
            )
        else:
            raise ValueError(
                f"Invalid mutual information method: {mutual_information_method}"
            )
