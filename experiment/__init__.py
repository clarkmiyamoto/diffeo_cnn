def load_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()

    print('Loaded model!')

    return model

def get_activation(model, layer_num: int, features):
    # Construct Hook
    intermediate_outputs = []
    def hook(module, input, output):
        intermediate_outputs.append(output)
    model.blocks[layer_num].register_forward_hook(hook)

    # Run Model-- data goes to hook
    with torch.no_grad():
        output = model(images)
        del output

    # Return output of hidden layer
    return intermediate_outputs[0]