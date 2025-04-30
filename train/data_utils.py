            if 'd_loc' in arguments.Config["model"]["name"]:
                sd = torch.jit.load(arguments.Config["model"]["path"])
                # sd is RecursiveScriptModule, load it into ori_model.model
                model_ori.model.load_state_dict(sd.state_dict())
                return model_ori
            elif 'LARD' in arguments.Config["model"]["name"]:
                _sd = torch.load(arguments.Config["model"]["path"], map_location=torch.device('cpu')).state_dict()
                # add prefix 'conv' to keys starting with 0/2/4, and 'linear' to keys starting with 7/9/11
                sd = {'conv' + k if k.startswith(('0', '2', '4')) else 'linear' + k if k.startswith(('7', '9', '11')) else k: v for k, v in _sd.items()}
                # sd is RecursiveScriptModule, load it into ori_model.model
                model_ori.model.load_state_dict(sd)
                return model_ori
            elif 'GTSRB' in arguments.Config["model"]["name"]:
                sd = torch.load(arguments.Config["model"]["path"], map_location=torch.device('cpu'))
                model_ori.model.load_state_dict(sd)
                return model_ori