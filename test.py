# -*- coding: utf-8 -*-
#!/usr/bin/python
import utils.miscUtils as miscUtils


# Datos de entrada del dataset
dataBase = "AR"
rootPath = miscUtils.getDataBasePath(dataBase)
cantFotos = miscUtils.photosPerPerson(rootPath)

print cantFotos