import io


def model_summary(model, count_params=True) -> str:
    print_buffer = io.StringIO()
    print(model, file=print_buffer)

    value = print_buffer.getvalue()
    return F'{value}\nParameters: {count_parameters(model)}' if count_params else value


def print_model_summary(model, count_params=True):
    print(model_summary(model, count_params))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
