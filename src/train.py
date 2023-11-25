from tqdm import tqdm

# Train function
def train(epoch, pair_loader, model, processor, loss_fn, optimizer, device):

    model.train()

    running_loss = 0
    for _, (data, targets) in enumerate(tqdm(pair_loader)):

        (text1, image1), (text2, image2) = data
    
        image1 = image1.to(device)
        image2 = image2.to(device)

        targets = targets.to(device)
        
        inputs1 = processor(text=text1, images=image1, return_tensors="pt", padding=True, truncation=True)
        inputs2 = processor(text=text2, images=image1, return_tensors="pt", padding=True, truncation=True)

        # Move tensors to the device
        inputs1 = {key: value.to(device) for key, value in inputs1.items()}
        inputs2 = {key: value.to(device) for key, value in inputs2.items()}



        optimizer.zero_grad()

        outputs1, outputs2 = model(inputs1, inputs2)

        loss = loss_fn(outputs1, outputs2, targets)

        running_loss += loss.item() * targets.size(0) # smaller batches count less

        loss.backward()
        optimizer.step()
        
        
    running_loss /= len(pair_loader.dataset)
    
    return running_loss
    
