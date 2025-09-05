import httpx

async def call_external_api(url: str, body: dict, params: dict = None, header: dict = None):
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=body, params=params, headers=header)
        return response.json()
    
async def call_externale_api_get(url:str,header:dict=None):
    async with httpx.AsyncClient() as client:
        response=await client.get(url,headers=header)
        return response.json()
<<<<<<< HEAD
    
=======
>>>>>>> 11b357a (python sources)

    
