from models.base_model import BaseModel


def create_model(opt) -> BaseModel:
    print(opt.model)
    if opt.model == 'gea_gan':
        from .gea_gan_model import GeaGanModel
        model = GeaGanModel(opt)
    elif opt.model == 'dea_gan':
        from .dea_gan_model import DeaGanModel
        model = DeaGanModel(opt)
    elif opt.model == 'time_predictor':
        from .time_predictor_model import TimePredictorModel
        model = TimePredictorModel(opt)
    elif opt.model == 'gea_TPN':
        from .gea_gan_model_TPN import GeaGanModelTPN
        model = GeaGanModelTPN(opt)
    elif opt.model == 'gea_DM':
        from .gea_gan_model_DM import GeaGanModelDM
        model = GeaGanModelDM(opt)
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    print("model [%s] was created" % (model.name()))
    return model
