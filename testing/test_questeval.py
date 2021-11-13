import utils
import numpy as np
import sys
import os
def run_questeval(references=[],candidates=[]):
    """
    Parameters
    ----------
    task: there is many type of task from questeval: summarization, text2text, data2text
    no_cuda: True use cpu, False use gpu, if available

    """
    check_dependencies("questeval/requirements.txt")
    PATH_F="questeval/bart.txt"
    if(os.path.exists(PATH_F)):
        print("Writing scores on: " + PATH_F)
    else:
        exit()
    if not candidates and not references:
        candidates=utils.load_preds("bart_preds.txt")
        references=utils.load_preds("bart_labels.txt")
    from questeval.questeval_metric import QuestEval
    questeval = QuestEval()
    l=[]
    for i in range(0, len(references)):
        score = questeval.corpus_questeval(hypothesis=[candidates[i]],list_references=[[references[i]]])
        l.append(score["corpus_score"])
        f=open(PATH_F,"a")
        f.write(str(score["corpus_score"]))
        f.write("\n")
        f.close()
        print(i)
    
    print(np.mean(l))

def check_dependencies(r_file):
    print("Checking for dependencies...")
    import subprocess
    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
    filed=open(r_file,"r")
    content=filed.read()
    dep_list=content.split("\n")
    filed.close()
    for dep in dep_list:
        if not dep in installed_packages:
            script="pip install " + dep
            os.system(script)
    print("Dependencies installed!")