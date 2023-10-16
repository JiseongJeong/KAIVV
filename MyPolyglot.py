from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
    PeftConfig,
    PeftModel
)

import evaluate
import torch
import numpy as np
from tqdm import tqdm

BNB_CONFIG = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16
                        )

PROMPT_CONFIG = PromptEncoderConfig(task_type="SEQ_CLS", 
                                  num_virtual_tokens=20, 
                                  encoder_hidden_size=128)                                

def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy", "f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

class PolyglotForClassification :
    def __init__(self,
                 device = 'cuda',
                 base_model = 'beomi/KoAlpaca-Polyglot-5.8B') :
        if device == 'cuda' :
            self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        self.base_model = base_model

    def load_model(self,
                   num_labels = 2,
                   gradient_checkpointing = True,
                   bnb_config = BNB_CONFIG,
                   peft_config = PROMPT_CONFIG) :

        model = AutoModelForSequenceClassification.from_pretrained(
                    self.base_model,
                    quantization_config=bnb_config,
                    device_map="auto",
                    num_labels= num_labels
                )
        
        if gradient_checkpointing :
            model.gradient_checkpointing_enable()
        
        model = get_peft_model(model, peft_config)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eos_token_id

        self.model = model
        self.tokenizer = tokenizer
        model.print_trainable_parameters()

    def Tokenize(self, dataset,
                 column = 'dialogs') :
        """dataset = load_dataset('csv', 
                     data_files={'train' : 'path', 
                               'test' : 'path'})"""
        if not 'test' in dataset.keys() :
            raise Exception("dataset에는 train과 test가 있어야 함")
 
        def tokenize_function(input):
            outputs = self.tokenizer(input[column], 
                                truncation=False,
                                max_length=None,
                                return_tensors= None,
                                padding = True)
            return outputs  
        
        tokenized_datasets = dataset.map(
                            tokenize_function,
                            batched = False,
                            remove_columns = column
        )
        

        return tokenized_datasets


    def train(self, tokenized_datasets, output_dir, training_args) :
        """ training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=50,
            max_steps=TRAIN_STEPS,
            learning_rate=LEARNING_RATE,
            logging_steps=10,
            optim="paged_adamw_8bit",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=50,
            save_steps=50,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="tensorboard",
            eval_accumulation_steps = 2

        )
        """
        self.output_dir = output_dir
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer,
                                                padding=True,
                                                pad_to_multiple_of=8,
                                                return_tensors="pt")
        trainer = Trainer(
                    compute_metrics=compute_metrics,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["test"],
                    model=self.model,
                    args=training_args,
                    tokenizer=self.tokenizer,
                    data_collator=data_collator,
                    )
        
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.config.use_cache =False

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.trainer = trainer

class PolyglotForInfer:
    def __init__(self,
                 device = 'cuda') :
        if device == 'cuda' :
            self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    def load_model(self,
                    load_path,
                    num_labels = 2,
                    bnb_config = BNB_CONFIG) :
        config = PeftConfig.from_pretrained(load_path) 
        base_model = config.base_model_name_or_path
        model = AutoModelForSequenceClassification.from_pretrained(base_model,
                                                                    quantization_config=bnb_config,
                                                                    device_map="auto",
                                                                    num_labels=num_labels)
        self.model = PeftModel.from_pretrained(model, load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.config = config
        self.base_model = base_model

    def Tokenize(self, dataset,
                 column = 'dialogs') :
        """infer_dataset = Dataset.from_pandas(infer_data)"""

        def tokenize_function(input):
            outputs = self.tokenizer(input[column], 
                                truncation=False,
                                max_length=None,
                                return_tensors= None,
                                padding = True)
            return outputs  
        
        tokenized_datasets = dataset.map(
                            tokenize_function,
                            batched = False,
                            remove_columns = column
        )

        return tokenized_datasets

    def BinaryInfer(self,
                    tokenized_datasets,
                    classes = ['normal', 'fraud']) :
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        with torch.no_grad():
            total_loss = 0
            for input in tqdm(iter(tokenized_datasets)) :
                labels = input['labels']
                output = self.model(torch.tensor(input['input_ids']).unsqueeze(0), labels = torch.tensor(labels).unsqueeze(0))
                total_loss += output.loss
                _, prediction = torch.max(output.logits, 1)
                # 각 분류별로 올바른 예측 수를 모읍니다
                if labels == prediction:
                    correct_pred[classes[labels]] += 1
                total_pred[classes[labels]] += 1
        # 각 분류별 정확도(accuracy)를 출력합니다
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        self.correct_pred =correct_pred
        self.total_pred = total_pred
        

