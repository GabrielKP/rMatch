import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from recall_matrix import CONFIG, console, get_logger

log = get_logger(__name__)


class MIRM:
    def __init__(
        self,
        mutual_information_method: str,
        mutual_information_normalize: bool,
        model_name: str | None = None,
        device: str | None = None,
        y_instruction: str = "",
        x_instruction: str = "",
        x_instruction_no_y: str = "",
        debug: bool = False,
    ):
        """Initialize the Mutual Information Recall Matrix (MIRM) object.

        Parameters
        ----------
        mutual_information_method: str
            The method to use for computing the mutual information.
            Currently one of: "rj_given_ei", "ei_given_rj", "ei_given_r0_to_rj".
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

        if mutual_information_method not in [
            "rj_given_ei",
            "rj_given_e0_to_ei",
            "r0_to_rj_given_ei",
            "rj_given_r0_to_rj_minus_1_and_ei",
            "ei_given_rj",
            "ei_given_r0_to_rj",
        ]:
            raise ValueError(
                f"Invalid mutual information method: {mutual_information_method}"
            )
        log.info(f"Using mutual information method: {mutual_information_method}")
        self.mutual_information_method = mutual_information_method

        log.info(f"Normalize mutual information: {mutual_information_normalize}")
        self.mutual_information_normalize = mutual_information_normalize

        if model_name is None:
            self.model_name: str = CONFIG.get("MODEL_NAME", "gpt2")  # type: ignore
        else:
            self.model_name = model_name
        log.info(f"Initializing model: {self.model_name}")

        if device is None:
            device = CONFIG.get("DEVICE", "auto")

        token = CONFIG.get("HF_TOKEN")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, token=token, device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=token, device_map=device
        )

        # get device
        self.device: torch.device = (
            next(iter(self.model.hf_device_map.values()))  # type: ignore
            if hasattr(self.model, "hf_device_map")
            else self.model.device
        )
        log.info(f"Using device: {self.device}")

        # test whether tokenizer adds a bos token
        # only reliable way seems to test this if not specified
        if hasattr(self.tokenizer, "add_bos_token"):
            self.add_bos_token = self.tokenizer.add_bos_token
        elif self.tokenizer.bos_token is None:
            self.add_bos_token = False
        else:
            tokens = self.tokenizer.encode("test")
            self.add_bos_token = tokens[0] == self.tokenizer.bos_token_id
        log.info(f"Tokenizer adds a bos token: {self.add_bos_token}")

        self.y_instruction = y_instruction
        self.x_instruction = x_instruction
        self.x_instruction_no_y = x_instruction_no_y

        self.debug = debug
        log.info(f"Debug mode: {self.debug}")

    def cross_entropy_x_given_y(
        self,
        input_str_x: str,
        input_str_y: str,
        print_str: str,
        prefix_x: str = "",
        verbose: bool = False,
        token_wise: bool = False,
    ) -> float:
        """Compute cross_entropy(LLM(X | Y), X)"""

        # tokenize
        if len(input_str_y) > 0:
            y_instruction = self.y_instruction
            x_instruction = self.x_instruction
            input_str = (
                f"{y_instruction}{input_str_y}{x_instruction}{prefix_x}{input_str_x}"
            )
            idx_char_x = (
                len(y_instruction)
                + len(input_str_y)
                + len(x_instruction)
                + len(prefix_x)
            )
        else:
            x_instruction_no_y = self.x_instruction_no_y
            input_str = f"{x_instruction_no_y}{prefix_x}{input_str_x}"
            idx_char_x = len(x_instruction_no_y) + len(prefix_x)

        batch_encoding = self.tokenizer(
            input_str, return_tensors="pt", return_offsets_mapping=True
        )
        input_ids: torch.Tensor = batch_encoding.input_ids
        offset_mapping: torch.Tensor = batch_encoding.offset_mapping

        idx_id_x = min(np.where(offset_mapping[:, :, -1] > idx_char_x)[1])

        # if tokenizer does not add bos token, we need to start from second x token
        if not self.add_bos_token:
            idx_id_x += 1

        with torch.no_grad():
            (logits, _) = self.model(
                input_ids=input_ids.to(self.device), return_dict=False
            )

        logits = logits.cpu()
        # choose only the logits from x

        # if self.add_bos_token == True, then shift arrows ^ by 1 to the right
        # input_ids: [... ,      y_j,      x_0,      x_1, ...,    x_n-1,      x_n]
        # labels_x :                        ^                                   ^
        labels_x = input_ids[:, idx_id_x:]
        # logits   : [... , pred_x_0, pred_x_1, pred_x_2, ..., pred_x_n, pred_x_n+1]
        # logits_x :             ^                                    ^
        logits_x = logits[:, idx_id_x - 1 : -1, :].cpu()

        logits_x_flat = logits_x.view(-1, logits_x.shape[-1])
        labels_x_flat = labels_x.view(-1)

        cross_entropy = F.cross_entropy(logits_x_flat, labels_x_flat, reduction="none")
        sum_cross_entropy = cross_entropy.numpy().sum()

        if verbose:
            prefix = f"[italic red]{print_str}={sum_cross_entropy:.3f}[/italic red]"
            decode_str_y = self.tokenizer.decode(input_ids[0, :idx_id_x])
            decode_str_x = self.tokenizer.decode(input_ids[0, idx_id_x:])

            console.print(
                f"{prefix} [white]{decode_str_y}[/white][green]{decode_str_x}[/green]"
            )
            if token_wise or self.debug:
                token_strs: list[str] = []
                for idx_token in range(len(labels_x_flat)):
                    tok = self.tokenizer.decode(labels_x_flat[idx_token])
                    ce = cross_entropy[idx_token].item()
                    token_strs.append(f"{tok}: {ce:.3f}")
                console.print(f"Token-wise cross entropy: {' | '.join(token_strs)}")

        return sum_cross_entropy

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
                        (f"> MI(E_{i}, R_{j}) = {recall_matrix[i, j]:.3f}"),
                        style="bold yellow",
                    )
            if self.debug:
                input("Press Enter to continue...")

        return recall_matrix

    def recall_matrix_rj_given_e0_to_ei(
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
                    input_str_y=" ".join(story_segments[: i + 1]),
                    print_str=f"H(R_j | E_0, ..., E_{i})",
                    verbose=verbose,
                )

                H_R_j = self.cross_entropy_x_given_y(
                    input_str_x=R_j,
                    input_str_y=" ".join(story_segments[:i]),
                    print_str=f"H(R_j | E_0, ..., E_{i - 1})",
                    verbose=verbose,
                )

                if normalize:
                    recall_matrix[i, j] = max(0, 1 - (H_R_j_given_E_i / H_R_j))
                else:
                    recall_matrix[i, j] = max(0, (H_R_j - H_R_j_given_E_i))
                if verbose:
                    console.print(
                        (f"> MI(E_0, ..., E_{i}, R_{j}) = {recall_matrix[i, j]:.3f}"),
                        style="bold yellow",
                    )
            if self.debug:
                input("Press Enter to continue...")

        return recall_matrix

    def recall_matrix_r0_to_rj_given_ei(
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
                    input_str_x=" ".join(recall_segments[: j + 1]),
                    input_str_y=E_i,
                    print_str=f"H(R_0 to R_{j} | E_{i})",
                    verbose=verbose,
                )

                H_R_j = self.cross_entropy_x_given_y(
                    input_str_x=" ".join(recall_segments[: j + 1]),
                    input_str_y="",
                    print_str=f"H(R_0 to R_{j})",
                    verbose=verbose,
                )

                if normalize:
                    recall_matrix[i, j] = max(0, 1 - (H_R_j_given_E_i / H_R_j))
                else:
                    recall_matrix[i, j] = max(0, (H_R_j - H_R_j_given_E_i))
                if verbose:
                    console.print(
                        (f"> MI(E_{i}, R_0 to R_{j}) = {recall_matrix[i, j]:.3f}"),
                        style="bold yellow",
                    )
            if self.debug:
                input("Press Enter to continue...")

        return recall_matrix

    def recall_matrix_rj_given_r0_to_rj_minus_1_and_ei(
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
                H_R_0_to_R_j_given_E_0_to_E_i = self.cross_entropy_x_given_y(
                    input_str_x=R_j,
                    input_str_y=E_i,
                    print_str=f"H(R_{j} | R_0 to R_{j - 1}, E_{i})",
                    prefix_x=" ".join(recall_segments[:j]) + " ",
                    verbose=verbose,
                )

                H_R_0_to_R_j_given_E_0_to_E_i_minus_1 = self.cross_entropy_x_given_y(
                    input_str_x=R_j,
                    input_str_y="",
                    print_str=f"H(R_{j} | R_0 to R_{j - 1})",
                    prefix_x=" ".join(recall_segments[:j]) + " ",
                    verbose=verbose,
                )

                if normalize:
                    recall_matrix[i, j] = max(
                        0,
                        1
                        - (
                            H_R_0_to_R_j_given_E_0_to_E_i
                            / H_R_0_to_R_j_given_E_0_to_E_i_minus_1
                        ),
                    )
                else:
                    recall_matrix[i, j] = max(
                        0,
                        (
                            H_R_0_to_R_j_given_E_0_to_E_i_minus_1
                            - H_R_0_to_R_j_given_E_0_to_E_i
                        ),
                    )
                if verbose:
                    console.print(
                        (
                            f"> MI(E_0 to E_{i}, R_0 to R_{j})"
                            f" = {recall_matrix[i, j]:.3f}"
                        ),
                        style="bold yellow",
                    )
            if self.debug:
                input("Press Enter to continue...")

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
                        (f"> MI(E_{i}, R_{j}) = {recall_matrix[i, j]:.3f}"),
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
                        (f"> MI(E_{i}, R_{j}{given_str}) = {recall_matrix[i, j]:.3f}"),
                        style="bold yellow",
                    )

        return recall_matrix

    def compute_mutual_information_recall_matrix(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        mutual_information_normalize: bool | None = None,
        mutual_information_method: str | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute the mutual information recall matrix

        Can override the mutual information method by passing it as an argument."""

        if mutual_information_method is None:
            mutual_information_method = self.mutual_information_method

        if mutual_information_normalize is None:
            mutual_information_normalize = self.mutual_information_normalize

        if mutual_information_method == "rj_given_ei":
            return self.recall_matrix_rj_given_ei(
                mutual_information_normalize,
                story_segments,
                recall_segments,
                verbose,
            )
        elif mutual_information_method == "rj_given_e0_to_ei":
            return self.recall_matrix_rj_given_e0_to_ei(
                mutual_information_normalize,
                story_segments,
                recall_segments,
                verbose,
            )
        elif mutual_information_method == "r0_to_rj_given_ei":
            return self.recall_matrix_r0_to_rj_given_ei(
                mutual_information_normalize,
                story_segments,
                recall_segments,
                verbose,
            )
        elif mutual_information_method == "rj_given_r0_to_rj_minus_1_and_ei":
            return self.recall_matrix_rj_given_r0_to_rj_minus_1_and_ei(
                mutual_information_normalize,
                story_segments,
                recall_segments,
                verbose,
            )
        elif mutual_information_method == "ei_given_rj":
            return self.recall_matrix_ei_given_rj(
                mutual_information_normalize,
                story_segments,
                recall_segments,
                verbose,
            )
        elif mutual_information_method == "ei_given_r0_to_rj":
            return self.recall_matrix_ei_given_r0_to_rj(
                mutual_information_normalize,
                story_segments,
                recall_segments,
                verbose,
            )
        else:
            raise ValueError(
                f"Invalid mutual information method: {mutual_information_method}"
            )
