import torch
from transformer import GPTModel
import numpy as np
import tiktoken
import torch.nn.functional as F

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)

    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)                                    
    else:
        num_batches = min(num_batches, len(data_loader))               
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()                                     
        else:
            break
    return total_loss / num_batches   

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []                        #A
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):                                                 #B
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                                                   #C
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                                                         #D
            optimizer.step()                                                        #E
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:                                        #F
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(                                                  #G
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()                #A
    with torch.no_grad():       #B
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

#A 评估阶段禁用 dropout，以确保结果稳定、可复现
#B 禁用梯度跟踪，减少计算开销

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " ")) # Compact print format
    model.train()


def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx)
        logits = logits[:, -1, :]
        if top_k:
            topk_logits, _ = torch.topk(logits, top_k)
            min_val = topk_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        
        if temperature > 0.0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probas, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        if next_token == eos_id:
            break
            
        idx = torch.cat((idx, next_token), dim=1)
    return idx

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])               #A
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):                                       #B
        q_w, k_w, v_w = np.split(                                                #C
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_block[b].att.query_w.weight = assign(
            gpt.trf_block[b].att.query_w.weight, q_w.T)
        gpt.trf_block[b].att.key_w.weight = assign(
            gpt.trf_block[b].att.key_w.weight, k_w.T)
        gpt.trf_block[b].att.value_w.weight = assign(
            gpt.trf_block[b].att.value_w.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_block[b].att.query_w.bias = assign(
            gpt.trf_block[b].att.query_w.bias, q_b)
        gpt.trf_block[b].att.key_w.bias = assign(
            gpt.trf_block[b].att.key_w.bias, k_b)
        gpt.trf_block[b].att.value_w.bias = assign(
            gpt.trf_block[b].att.value_w.bias, v_b)

        gpt.trf_block[b].att.proj_out.weight = assign(
            gpt.trf_block[b].att.proj_out.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_block[b].att.proj_out.bias = assign(
            gpt.trf_block[b].att.proj_out.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_block[b].ff.layers[0].weight = assign(
            gpt.trf_block[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_block[b].ff.layers[0].bias = assign(
            gpt.trf_block[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_block[b].ff.layers[2].weight = assign(
            gpt.trf_block[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_block[b].ff.layers[2].bias = assign(
            gpt.trf_block[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_block[b].norm1.scale = assign(
            gpt.trf_block[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_block[b].norm1.shift = assign(
            gpt.trf_block[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_block[b].norm2.scale = assign(
            gpt.trf_block[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_block[b].norm2.shift = assign(
            gpt.trf_block[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())