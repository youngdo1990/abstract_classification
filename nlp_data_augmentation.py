# -*- coding: utf-8 -*-
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import os
import argparse


def generate_files(source="test_data.xlsx", skiprows=2):
    return


if __name__ == "__main__":
    text = 'The quick brown fox jumps over the lazy dog .'
    aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="substitute")
    augmented_text = aug.augment(text, n=3)

    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text)