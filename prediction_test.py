

from predict import *
import pandas as pd
import logging
logging.basicConfig(level=logging.ERROR)

BATCH_SIZE = 3

def get_texts(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.iloc[1000:]
    titles, labels = df["Title"].tolist(), df["Conference"].tolist()
    return titles, labels


def cut_batches(text_list, label_list, batch_size=10):
    final_text_list = lambda text_list, batch_size: [text_list[i:i+batch_size] for i in range(0, len(text_list), batch_size)]
    texts = final_text_list(text_list, batch_size)
    final_label_list = lambda label_list, batch_size: [label_list[i:i+batch_size] for i in range(0, len(label_list), batch_size)]
    labels = final_label_list(label_list, batch_size)
    return texts, labels


if __name__ == "_main__":
    dataset_path = "./data/dataset.csv"

    texts, labels = get_texts(dataset_path)
    class_dict = get_class_dict()
    texts, labels = cut_batches(texts, labels)
    class_dict = get_class_dict() # 클래스 예측한 다음에 클래스 레이블을 클래스 이름으로 바꾸기
    model = inference_model(pretrained_model="./allenai-specter", class_dict=class_dict)

    # for text, label in zip(texts, labels):
    #     preds = predict(model=model, texts=text, class_dict=class_dict, batch_size=3)
    #     for p, l in zip(preds, label)