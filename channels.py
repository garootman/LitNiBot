channels = [
    {"from": -1001716081508, "to": 5566311336, 
     "rules": {"text": {"prompt": "Rewrite this in english:", "translate": "en"}, 
     "image": {"crop": 0.5, "rotate": 0.5, "watermark": "watermark2.png", "wm_loc": "c"}, 
     "video": {"watermark": "watermark2.png", "angle": 1.5, "noise": -5, "wm_loc": "br"}}
    }, 
    
    {"from": 62408647, "to": 2041631011, 
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