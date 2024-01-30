import torch
import torch.nn.functional as F

def cross_entropy(logits, target):
    "MPS crashes on cross entropy if passed single target tensor. I think this is because int64 is not supported"
    if target.ndim == 1:
        target = F.one_hot(target).to(torch.int32).float()
    return F.cross_entropy(logits, target)


#################################
## OLD

def accuracy(xb, yb):
    return (xb.argmax(1) == yb).float().mean()


def fit(epochs, model, loss_func, train_dl, valid_dl, opt):
    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        model.train()
        for xb, yb in train_dl:
            logits = model(xb)
            loss = loss_func(logits, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            vloss = []
            vacc = []
            for xb, yb in valid_dl:
                vlogits = model(xb)
                vloss.append(loss_func(vlogits, yb))
                vacc.append(accuracy(vlogits, yb))

            print(
                f"Val Loss: {sum(vloss)/len(vloss):.4f} Val Accuracy: {sum(vacc)/len(vacc):.4f}"
            )

