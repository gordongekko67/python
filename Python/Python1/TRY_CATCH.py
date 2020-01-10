def try_except_except_test(x):
    try:
        n = int(x)  # prova a convertire x in intero
    except ValueError:
        # eseguito in caso di ValueError
        print('Invalid number!')
    except TypeError:
       # eseguito in caso di TypeError
        print('Invalid type!')
    else:
         # eseguito se non ci sono errori
        print('Valid number!')


try_except_except_test(7)
