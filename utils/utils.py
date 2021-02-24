from prettytable import PrettyTable


def print_infos(model):
    table = PrettyTable(["Modules", "Parameters", "Grad"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            grad = "True"
        else:
            grad = "False"
        param = parameter.numel()
        table.add_row([name, param, grad])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
