
def get_input(message, options_messages, options):

    while True:
        print(message)
        for op in options_messages:
            print(op)
        net_index = input()
        try:
            option = options[int(net_index)]
            return option
        except (IndexError, ValueError):
            print("invalid input. please try again")

