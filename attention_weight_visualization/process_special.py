def remove_special_tokens(tokens, scores, special_tokens=None):
    if special_tokens is None:
        special_tokens = {"<s>", "</s>"}
    
    new_tokens = []
    new_scores = []

    for t, s in zip(tokens, scores):
        if t in special_tokens:
            continue
        new_tokens.append(t)
        new_scores.append(s)

    return new_tokens, new_scores

