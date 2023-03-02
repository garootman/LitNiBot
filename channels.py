channels = [
    {"from": -1001716081508, "to": [5566311336], 
     "shcedule":[
         {'days':[1,2,3,4,5], 'times':[("02:30","02:40"),("15:23","22:40")]}, # по будням с полуночи и с 15-23 до 22-40
         {'days':[6], 'times':[("16:30","23:59")]}, # в субботу с 16-3 по 23-59
         {'days':[7], 'times':[]}, # в воскресенье не берём

                ],
     "rules": {"text": {"prompt": "Rewrite this in english:", "translate": "en"}, 
     "image": {"crop": 0.5, "rotate": 0.5, "watermark": "watermark.png", "wm_loc": "c"}, 
     "video": {"watermark": "watermark.png", "angle": 1.5, "noise": -5, "wm_loc": "br"}}
    }, 
    
    {"from": 62408647, "to": [2041631011, 5566311336], 
     "rules": {"text": {"prompt": "Make a joke out this content:", "translate": "fr"}, 
     "image": {"crop": 1.0, "rotate": 0.4, "watermark": "watermark.png", "wm_loc": "tl"}, 
     "video": {"watermark": "watermark.png", "angle": 0.5, "noise": -10, "wm_loc": "c"}}
    }, 
###   wm_loc - расположение вотермарки. может быть одним из: 
# br - нижний правый угол
# tl - верхний левый угол
# tr - верхний правый
# bl - нижний левый
# c - центр, на всю картинку
    
# angle - угол поворота в градусах
# translate - можешь оставить пустым
#
    
]