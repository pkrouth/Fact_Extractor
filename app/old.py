
import uvicorn, aiohttp, asyncio

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from fastai import *
from fastai.text import *


app = Starlette()
app.debug = True
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory="app/static"))
model_file_url = 'https://www.dropbox.com/s/3y8ht35rxr0s1ao/model_exported.pkl?dl=0'
#model_file_name = 'model_exported.pkl'
path = Path("app")

async def setup_learner():
    #await download_file(model_file_url, path/'models'/f'{model_file_name}.pkl')
    #learn = cnn_learner(data_bunch, models.resnet34, pretrained=False)
    #learn = text_classifier_learner(data_bunch, AWD_LSTM, pretrained=False)
    #learn = get_t
    learn =load_learner(path/'models','model_exported.pkl')
    #learn.load(model)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()



@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    #img_bytes = await (data['file'].read())
    # Introduce a reader which can read .txt files
    #input_text = data
    #txt = open_image(BytesIO(img_bytes))
    ## Here there needs to be a function which outputs string
    #txt = open_text(input_text)#,encoding='utf-8') #use this for file
    class_, ind, probs = learn.predict(data['unique_name'])
    score = probs[ind].item()
    if score < 0.5:
        Confidence='Low'
    else:
        Confidence='High'
    return JSONResponse({'label': str(class_),'score': float(score), 'Confidence':Confidence}) # This remains same



if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8008)
