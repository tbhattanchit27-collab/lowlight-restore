# Deploying the app (Hugging Face Spaces, free CPU)

Takes about 15 minutes. You need a GitHub account and a Hugging Face account.

## 1. Push to GitHub
```
cd lowlight-restore
git init
git add .
git commit -m "P1 low-light restoration"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/low-light-restoration.git
git push -u origin main
```
The repo is about 90 MB (mostly `data/raw`). If GitHub complains, that is still fine:
no single file is over 100 MB.

## 2. Create the Space
1. huggingface.co -> New -> Space.
2. Name: `low-light-restoration`, SDK: **Gradio**, hardware: **CPU basic (free)**, Public.
3. Push the code to the Space (it is just another git remote):
```
git remote add space https://huggingface.co/spaces/YOUR-USERNAME/low-light-restoration
git push space main
```
   When asked for a password, use a Hugging Face **access token** with write rights
   (Settings -> Access Tokens).
   If the push is rejected for size, push without the data folder:
   `git rm -r --cached data results && git commit -m "app only" && git push space main`
   (do this on a separate branch if you want to keep data on GitHub).
4. The Space builds for 3–6 minutes. The README header (`sdk: gradio`, `app_file: app.py`)
   tells it how to start. The first request after a sleep takes ~20 s; after that each
   photo takes 1–3 s.

## 3. Check before judging
* Open the Space URL on a phone and a laptop.
* Upload a photo from your own phone taken in a dark room. Judges will do this.
* Try the "Compare all three" tab.
* Put the URL in README.md and in the submission form.

## Plan B
* Docker on any host (Render, Railway, a lab PC): `docker build -t lowlight . && docker run -p 7860:7860 lowlight`
* Emergency public link from a laptop: change the last line of `app.py` to
  `demo.queue().launch(share=True)` – gives a 72-hour gradio.live URL.
