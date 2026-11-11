import uvicorn
if __name__=='__main__': uvicorn.run('bunyaan_model.api:app',host='127.0.0.1',port=7860,reload=False)
