import pickle as pkl

identity_pkl = "./datasets/splits/identity_dict_test_split_smpl.pkl"
with open(identity_pkl, "rb") as f:
    identity_dict_smpl = pkl.load(f)


print(identity_dict_smpl["000000"].keys())
i = 0

for k, v in identity_dict_smpl.items():
    print(k)
    # print(v.keys())

    for k1, v1 in v.items():
        print(k1)
        # if isinstance(v1, str):
        #     print(v1)
        # else:
        #     print(v1.shape)

        if k1 == "betas_B":
            print(v1)
            print(v1.shape)

        if k1 == "betas_B_norm":
            print(v1)
            print(v1.shape)

    i += 1

    if i > 1:
        break
