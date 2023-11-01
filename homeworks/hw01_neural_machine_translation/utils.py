import torch


def flatten(l):
    return [item for sublist in l for item in sublist]


def remove_tech_tokens(mystr, tokens_to_remove=["<eos>", "<sos>", "<unk>", "<pad>"]):
    return [x for x in mystr if x not in tokens_to_remove]


def get_text(x, TRG_vocab):
    text = [TRG_vocab.itos[token] for token in x]
    try:
        end_idx = text.index("<eos>")
        text = text[:end_idx]
    except ValueError:
        pass
    text = remove_tech_tokens(text)
    if len(text) < 1:
        text = []
    return text


def generate_translation(src, trg, model, TRG_vocab):
    model.eval()

    trg_len, _ = trg.shape
    sos = trg[0, :].unsqueeze(0)
    output = model(src, sos).argmax(dim=-1)

    for _ in range(1, trg_len):
        output = model(src, torch.cat((sos, output), dim=0)).argmax(dim=-1)
    assert output.shape == trg.shape

    output = output.cpu().numpy()

    original = get_text(list(trg[:, 0].cpu().numpy()), TRG_vocab)
    generated = get_text(list(output[:, 0]), TRG_vocab)

    print("Original: {}".format(" ".join(original)))
    print("Generated: {}".format(" ".join(generated)))
    print()
