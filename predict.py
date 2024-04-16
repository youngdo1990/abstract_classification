# -*- coding:utf-8 -*-
from model import *


def get_class_dict():
    with open("label_dict", "r", encoding="utf-8", errors="ignore") as dict_file:
        txt = dict_file.read()
        label_dict = eval(txt)
    return {v: k for k, v in label_dict.items()}


def data_prepare(user_input, tokenaizer, batch_size):
    if type(user_input) != list:
        user_input = [user_input]
    
    tokenizer = BertTokenizer.from_pretrained(tokenaizer,
                                            do_lower_case=True)
    
    encoded_input = tokenizer.batch_encode_plus(
        user_input, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=256, 
        return_tensors='pt'
    )

    input_ids = encoded_input['input_ids']
    attention_masks = encoded_input['attention_mask']
    labels = torch.tensor([0]*len(encoded_input['input_ids']))
    data = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(data, 
                            sampler=SequentialSampler(data), 
                            batch_size=batch_size)
    return dataloader


def inference_model(pretrained_model, class_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_load(pretrained_model=pretrained_model, label_dict=[0]*len(class_dict))
    model.to(device)
    model.load_state_dict(torch.load('data_volume/finetuned_BERT_epoch_5.model', map_location=torch.device('cpu')))
    return model


def predict(model, texts, class_dict, batch_size=3):
    dataloader = data_prepare(user_input=texts, tokenaizer="./allenai-specter", batch_size=batch_size)
    preds = evaluate(model=model, dataloader_val=dataloader, predict=True)
    preds = [class_dict[p] for p in preds]
    return preds
    

if __name__ == "__main__":
    class_dict = get_class_dict() # 클래스 예측한 다음에 클래스 레이블을 클래스 이름으로 바꾸기
    model = inference_model(pretrained_model="./allenai-specter", class_dict=class_dict)

    texts = ["FaceKit: A Database Interface Design Toolkit.",
            "How Best to Build Web-Scale Data Managers? A Panel Discussion.",
            "TELEIOS: A Database-Powered Virtual Earth Observatory.",
            "Capturing Global Transactions from Multiple Recovery Log Files in a Partitioned Database System.",
            "When Speed Has a Price: Fast Information Extraction Using Approximate Algorithms.",
            "Specifying and Enforcing Intertask Dependencies.",
            "Features of a Conceptual Schema.",
            "On Semantic Issues Connected with Incomplete Information Data Bases (Abstract).",
            "Information Processing for CAD/VLSI on a Generalized Data Management System.",
            "Slicing Long-Running Queries."
            ]

    predict(model, texts, class_dict, batch_size=3)