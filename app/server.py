
import uvicorn, aiohttp, asyncio

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from fastai import *
from fastai.text import *
from nltk.tokenize import sent_tokenize
from custom_functions import *
app = Starlette()
app.debug = True
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory="app/static"))
#model_file_url = 'https://www.dropbox.com/s/3y8ht35rxr0s1ao/model_exported.pkl'
#model_file_name = 'model_exported'

path = Path(__file__).parent

async def setup_learner():
    #await download_file(model_file_url, path/'models'/f'{model_file_name}.pkl')
    learn_fwd =load_learner(path/'models','model_fwd.pkl')
    learn_bwd =load_learner(path/'models','model_bwd.pkl')
    classes_ = ['FACTS','Non-FACTS']
    return learn_fwd, learn_bwd, classes_

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn_fwd, learn_bwd, classes_ = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    case = data['unique_name']
    predictor = Predictor(casetext=case,model_fwd=learn_fwd,model_bwd=learn_bwd,labels=classes_, sent_tokenize=sent_tokenize)
    output = predictor.filter_facts()
    #score = probs[ind].item()
    #if score < 0.5:
    #    Confidence='Low'
    #else:
    #    Confidence='High'
    #print(case)
    #return JSONResponse({'label': str(classes_[0]),'score': float(0.5), 'output':case}) # This remains same
    return JSONResponse({'output': str(output)}) # This remains same

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8008)
