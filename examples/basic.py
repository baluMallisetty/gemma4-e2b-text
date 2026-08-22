from gemma4_e2b_text import Gemma4


with Gemma4(backend="cpu") as model:
    answer = model.generate(
        "Florida is a state in which country? Answer in one word."
    )
    print(answer)
