import pandas as pd
class Predictor:
    def __init__(self,casetext=None,model_fwd=None,model_bwd=None,labels=None,sent_tokenize=None):
        self.casetext = casetext
        self.model_fwd = model_fwd
        self.model_bwd = model_bwd
        self.fact_labels = labels
        self.sent_tokenize = sent_tokenize

    def fact_predict(self,text=None,fwd=None,bwd=None,verbose=False,labels = None,topn=1):
        top_label_fwd, top_label_ind_fwd, preds_fwd = self.model_fwd.predict(text)
        top_label_bwd, top_label_ind_bwd, preds_bwd = self.model_bwd.predict(text)
        #set_trace()
        assert preds_fwd.sum().item()>0.99
        assert preds_bwd.sum().item()>0.99
        preds = (preds_fwd+preds_bwd)/2
        #[[labels[i] for i in preds.topk(topn)[1]],top_label_fwd,top_label_bwd]
        return any([str(top_label_bwd)=='FACTS',str(top_label_fwd)=='FACTS'])

    def flatten_list(self,list):
        case_flatten = []
        for sent in list:
            if isinstance(sent,type(' ')):
                case_flatten.append(sent)
            else:
                case_flatten = case_flatten+self.flatten_list(sent)
        return case_flatten

    def process_text(self):
        case_list = [self.sent_tokenize(sent) for sent in self.casetext.split('\n')]
        case_flatten = self.flatten_list(case_list)
        single_case = pd.DataFrame(enumerate(case_flatten),columns=['sentence_id','sentence_text'])
        single_case['is_fact'] = single_case.sentence_text.apply(self.fact_predict)
        return single_case

    def filter_facts(self,word_threshold=5):
        result = self.process_text()
        result['length'] = result.sentence_text.apply(lambda x: len(x.split()))
        output = result[(result.length>word_threshold) & (result.is_fact)].sentence_text.get_values()
        return '\n\n '.join(output)
