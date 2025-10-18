"""This script runs specified mutual information method to compute
the recall matrix for the first 10 cyoa participants."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from recall_matrix import console
from recall_matrix.mutual_information_recall_matrix import MIRM


def playground(model_name: str | None = None):
    y_instruction = (
        "Your task is to predict what a human would recall about a"
        " story segment."
        "\nThe following is the story segment:\n"
    )
    x_instruction = "\nThe following is the matching recall segment:\n"
    x_instruction_no_y = (
        "Your task is to predict what a human would recall about a"
        " story segment."
        "\nThe following is the story segment:\nNo story segment."
        "\nThe following is the matching recall segment:\n"
    )

    # 2. init mirm
    mirm = MIRM(
        model_name=model_name,
        mutual_information_method="rj_given_ei",
        mutual_information_normalize=False,
        y_instruction=y_instruction,
        x_instruction=x_instruction,
        x_instruction_no_y=x_instruction_no_y,
    )

    inputs = [
        (
            """Blackness. You're slowly waking from a dream. What was it about? It is already beginning to fade… Was there a mirror? A rabbit? I was having tea with someone… It all seemed so urgent! You try grasping for details, to try and hold onto the dream for just a little bit longer. Fading. It slips through your fingers and disappears. Like dreams often do. Slowly, whirling colors and shapes appear.""",  # noqa: E501
            """A rabbit is jumping around.""",
        ),
        (
            """Blackness. You're slowly waking from a dream. What was it about? It is already beginning to fade… Was there a mirror? A rabbit? I was having tea with someone… It all seemed so urgent! You try grasping for details, to try and hold onto the dream for just a little bit longer. Fading. It slips through your fingers and disappears. Like dreams often do. Slowly, whirling colors and shapes appear.""",  # noqa: E501
            """You wake up from a dream.""",
        ),
        (
            """Blackness. You're slowly waking from a dream. What was it about? It is already beginning to fade… Was there a mirror? A rabbit? I was having tea with someone… It all seemed so urgent! You try grasping for details, to try and hold onto the dream for just a little bit longer. Fading. It slips through your fingers and disappears. Like dreams often do. Slowly, whirling colors and shapes appear.""",  # noqa: E501
            """You woke up from a dream.""",
        ),
        (
            """Blackness. You're slowly waking from a dream. What was it about? It is already beginning to fade… Was there a mirror? A rabbit? I was having tea with someone… It all seemed so urgent! You try grasping for details, to try and hold onto the dream for just a little bit longer. Fading. It slips through your fingers and disappears. Like dreams often do. Slowly, whirling colors and shapes appear.""",  # noqa: E501
            """Woke up from what felt like a dream.""",
        ),
        (
            """Blackness. You're slowly waking from a dream. What was it about? It is already beginning to fade… Was there a mirror? A rabbit? I was having tea with someone… It all seemed so urgent! You try grasping for details, to try and hold onto the dream for just a little bit longer. Fading. It slips through your fingers and disappears. Like dreams often do. Slowly, whirling colors and shapes appear.""",  # noqa: E501
            """And the guy got pied.""",
        ),
        (
            """And he stops and looks at me and he says, “Listen up, punk.” And right then, there’s a blur in the corner of my eye which becomes this figure holding a cream pie which becomes the guy standing next to me mashing a cream pie into Dean McGowan’s face.""",  # noqa: E501
            """You woke up from a dream.""",
        ),
        (
            """And he stops and looks at me and he says, “Listen up, punk.” And right then, there’s a blur in the corner of my eye which becomes this figure holding a cream pie which becomes the guy standing next to me mashing a cream pie into Dean McGowan’s face.""",  # noqa: E501
            """A rabbit is jumping around.""",
        ),
        (
            """And he stops and looks at me and he says, “Listen up, punk.” And right then, there’s a blur in the corner of my eye which becomes this figure holding a cream pie which becomes the guy standing next to me mashing a cream pie into Dean McGowan’s face.""",  # noqa: E501
            """And the guy got pied.""",
        ),
    ]

    for input_y, input_x in inputs:
        H_R_j_given_E_i = mirm.cross_entropy_x_given_y(
            input_str_x=input_x,
            input_str_y=input_y,
            print_str="H(R_j | E_i)",
            verbose=True,
            token_wise=True,
        )

        H_R_j = mirm.cross_entropy_x_given_y(
            input_str_x=input_x,
            input_str_y="",
            print_str="H(R_j)",
            verbose=True,
            token_wise=True,
        )

        MI = max(0, (H_R_j - H_R_j_given_E_i))
        console.print(f"MI(R_j, E_i) = {MI:.3f}", style="bold yellow")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-mn", "--model_name", type=str, default=None)
    args = parser.parse_args()
    playground(model_name=args.model_name)
