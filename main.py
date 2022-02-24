from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from src import scs


app = FastAPI(title="SCD")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), "static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process", response_class=HTMLResponse)
def process(
    request: Request,
    signal: str = Form(...),
    kernel: str = Form(...),
    p: int = Form(...),
    q: float = Form(...),
) -> dict:
    """
    Calculate the result of a normal convolution and a sharpened cosine similarity
    on the input signal (1D), given the kernel/filter and hyperparameters of SCS.
    """
    signal = [float(s) for s in signal.split(",")]
    kernel = [float(k) for k in kernel.split(",")]
    result = scs(signal=signal, kernel=kernel, p=p, q=q)

    # adjust to normalize graph heights
    max_conv = max([abs(r) for r in result["convolution"]])
    result["convolution"] = [v / max_conv for v in result["convolution"]]

    # add signal and kernel data
    max_signal = max([abs(s) for s in signal])
    result["signal"] = [s/max_signal for s in signal]
    result["kernel"] = kernel

    max_kernel = max([abs(k) for k in kernel])
    result["kernel"] = [k/max_kernel for k in kernel]

    return templates.TemplateResponse(
        "result.html", {"request": request, "result": result}
    )
