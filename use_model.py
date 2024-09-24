from transformers import GPT2LMHeadModel, GPT2Tokenizer


model = GPT2LMHeadModel.from_pretrained("./resultados", low_cpu_mem_usage=True)


tokenizer = GPT2Tokenizer.from_pretrained("./resultados")
tokenizer.pad_token = "<pad>"


profissoes_info = {}
with open('professions.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if ':' in line:
            profissao, info = line.split(':', 1)
            profissao = profissao.strip()
            info = info.strip()
            profissoes_info[profissao.lower()] = info

def gerar_texto(profissao):
    info = profissoes_info.get(profissao.lower())
    if info:
        return f"{profissao.capitalize()}:\n{info}"
    else:
        prompt = f"Quais são as responsabilidades e habilidades de um {profissao}?"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        try:
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=150,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return f"{profissao.capitalize()}:\n{generated_text}"
        except Exception as e:
            return f"Erro ao gerar texto: {str(e)}"

if __name__ == "__main__":
    profissao_usuario = input("Digite sua profissão: ")
    resultado = gerar_texto(profissao_usuario)
    print(resultado)
