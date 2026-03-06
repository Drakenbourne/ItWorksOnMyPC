# Run On Google Colab (GPU)

## 1. Start Colab with GPU
- Open Colab
- Runtime -> Change runtime type -> Hardware accelerator: `GPU`

## 2. Clone your repo
```bash
git clone https://github.com/jitgoel/BMSCE-XCEL-TS100.git
cd BMSCE-XCEL-TS100/ItWorksOnMyPC/source/ai-interview-bot
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
pip install pyngrok
```

## 4. Start Streamlit in background
```bash
!streamlit run app.py --server.port 8501 --server.headless true &
```

## 5. Expose public URL
```python
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(public_url)
```

Open the printed URL in browser.

## Notes
- This project auto-detects GPU and uses it when available.
- First run still includes model download time.
- Keep the Colab session active during demo.