# from fastapi import FastAPI, File, UploadFile, HTTPException
# from typing import List
# import easyocr
# from PIL import Image
# import numpy as np
# import cv2
# from numpy import asarray 
# from io import BytesIO
# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Create FastAPI app
# app = FastAPI()

# def extract_text_from_image(imagepath):
  
   
  
#     result = reader.readtext(imagepath)
#     result
#     final=''
#     for i in range(len(result)):

#         final=final+result[i][1]

#     return final

# score_list = {'Whole Wheat Flour (Atta) (32.7%)':80,
#              ' Refined Wheat Flour (Maida) (27.3%)':60,
#              ' Sugar':30,
#              ' Gluten':50,
#               ' Yeast':40,
#               ' Wheat Bran':95,
#               ' lodized Salt':60,
              
             
#              }

# def score_generator(ingridients):
#     ing_list = ingridients.split(",")
    
#     sum = 0
#     counter=0
#     for ing in ing_list:
#         #print(ing)
#         if ing in score_list.keys():
#             #print(ing,score_list[ing])
#             counter+=1
#             sum+=score_list[ing]
            
#     score = sum/counter
#     temp=round(score,2)
#     return temp     
    
# @app.get("/")
# def home():
#     return {"health_check":"ok"}

# @app.post("/extract_text/")
# async def extract_text(image: UploadFile = File(...)):
#     content=await image.read()
#     img=Image.open(BytesIO(content))
#     img.save('sample.jpg')
#     extracted_text=extract_text_from_image('sample.jpg')
#     score=score_generator(extracted_text)

    
#     return {"score": score,"extracted_text": extracted_text}

# # Run the FastAPI server using Uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import easyocr
import os
from PIL import Image
import numpy as np
import cv2
from numpy import asarray 
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException


os.environ["GOOGLE_API_KEY"] = "AIzaSyAZSP3AXxW8vmcalgAYmikudzSqT_IgWQk"
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Create FastAPI app
app = FastAPI()

def extract_text_from_image(imagepath):
  
   
  
    result = reader.readtext(imagepath)
    result
    final=''
    for i in range(len(result)):

        final=final+result[i][1]

    return final
#"D://vs code files//ing.png"



def getscore(text):

    title_template = PromptTemplate(
            input_variables=['ingredients'],
            template="""i need you to be a score predictor and rate these gives ingredients of the bread and give the rating on the basis of how healthy is the bread with these ingredients i want you to give rating from 100

    ingredients= {ingredients}
    just return me the overall score just the number nothing else"""
        )

    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='output')

    response = title_chain.invoke({'ingredients': text})
    
    return response["output"]
#print(text)

#text = extract_text_from_image("D://vs code files//ing.png")

#score = getscore(text)
#print(score)

@app.get("/")
def home():
    return {"health_check":"ok"}

@app.post("/extract_text/")
async def extract_text(image: UploadFile = File(...)):
    content=await image.read()
    img=Image.open(BytesIO(content))
    img = img.convert("RGB")


    img.save('sample.jpg')

    extracted_text=extract_text_from_image('sample.jpg')
    score=getscore(extracted_text)

    
    return {"score": score,"extracted_text": extracted_text}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

