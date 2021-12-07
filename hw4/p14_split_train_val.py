# P14
# s_train = ""
# s_val = ""
# with open('hw4_train_trans.dat', 'r') as f:
#     for i, line in enumerate(f.readlines()):
#         if (i < 120):
#             s_train += line
#         else:
#             s_val += line

# print(len(s_train.split('\n')))
# print(len(s_val.split('\n')))

# with open('hw4_p14_train.dat', 'w') as f:
#     f.write(s_train)
# with open('hw4_p14_val.dat', 'w') as f:
#     f.write(s_val)


### P16
n_fold = 5
with open('hw4_train_trans.dat', 'r') as f:
    lines = f.readlines()
    for fold in range(n_fold):
        s_train = ""
        s_val = ""
        for i, line in enumerate(lines):
            if (i >= fold*40 and i < (fold+1)*40):
                s_val += line                
            else:
                s_train += line
        print(len(s_train.split('\n')))
        print(len(s_val.split('\n')))

        with open(f'hw4_p16_train_{fold}_fold.dat', 'w') as f:
            f.write(s_train)
        with open(f'hw4_p16_val_{fold}_fold.dat', 'w') as f:
            f.write(s_val)
