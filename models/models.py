from models.base_model import BaseModel


def create_model(opt) -> BaseModel:
    model = None
    print(opt.model)
    if opt.model == 'gea_gan':
        from .gea_gan_model import GeaGanModel
        model = GeaGanModel()
    elif opt.model == 'dea_gan':
        from .dea_gan_model import DeaGanModel
        model = DeaGanModel()
    elif opt.model == 'time_predictor':
        from .time_predictor_model import TimePredictorModel
        model = TimePredictorModel()
    elif opt.model == 'gea_TPN':
        from .gea_gan_model_TPN import GeaGanModelTPN
        model = GeaGanModelTPN()
    elif opt.model == 'gea_DM':
        from .gea_gan_model_DM import GeaGanModelDM
        model = GeaGanModelDM()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
