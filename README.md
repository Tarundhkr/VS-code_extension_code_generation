Code Generation API with FastAPI

This project sets up a FastAPI server that provides code generation capabilities using a fine-tuned Salesforce CodeT5 model. The server is hosted on Google Colab and can be accessed via a public URL generated using ngrok.
Features

    Code Generation: Generate code snippets based on input text using a state-of-the-art language model.
    GPU Acceleration: Utilizes CUDA for faster inference if a compatible GPU is available.

Prerequisites

Before you begin, ensure you have the following:

    Google Colab account
    Access to ngrok (for tunneling)

Installation

Clone this Repository:

bash
    
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo

Install Required Packages:

Run the following command to install the necessary packages:

bash

    !pip install fastapi uvicorn ngrok colabcode transformers torch

Usage

Setup and Run:

Open a new Google Colab notebook.

Copy and paste the following code into a cell:

python

    from fastapi import FastAPI, Request
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    import ngrok
    import uvicorn
    import asyncio
    import nest_asyncio
    from google.colab import userdata

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = "Salesforce/codet5p-2b"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                                  torch_dtype=torch.float16,
                                                  trust_remote_code=True).to(device)

    nest_asyncio.apply()
    ngrok.set_auth_token(userdata.get('auth'))
    app = FastAPI()

    @app.post("/generate/")
    async def generate_text(request: Request):
        data = await request.json()
        input_text = data['text']
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs,
                                     max_length=75,
                                     min_length=50)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": output_text}

    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel)
    uvicorn.run(app, port=8000)

Access the API:

After running the code, you'll see a public URL printed in the output. This URL is the endpoint for your FastAPI server.

You can send POST requests to the /generate/ endpoint with a JSON body containing your input text.

json
    
    {
      "text": "Your input text here"
    }

The server will return a JSON response with the generated code:

json

        {
          "generated_text": "Generated code snippet here"
        }

Notes

    Ensure that you replace 'auth' in ngrok.set_auth_token(userdata.get('auth')) with your actual ngrok auth token.
    Adjust max_length and min_length parameters in the generate function as needed to fit your use case.
