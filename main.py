from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from translate import Translator
from kivy.uix.dropdown import DropDown
import numpy as np
from kivy.app import App
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivymd.app import MDApp
from kivy.uix.settings import SettingsWithTabbedPanel
from kivy.logger import Logger
from kivy.config import ConfigParser
import cv2
import pyttsx3


config = ConfigParser()
config.read('my.ini')
language = config.get('My Label', 'text')
scale = 0.00392
translator= Translator(to_lang=language)



def say_bitch(a, lang):
    tts = pyttsx3.init()

    voices = tts.getProperty('voices')

    # Задать голос по умолчанию

    tts.setProperty('voice', lang)

    # Попробовать установить предпочтительный голос

    if lang == 'ru':
        for voice in voices:
            if voice.name == 'Alexandr':
                tts.setProperty('voice', voice.id)
    else:
        for voice in voices:
            # if voice.name == 'Alexandr':
            tts.setProperty('voice', voice.id)
    tts.say(a)
    tts.runAndWait()

# Задать голос по умолчанию



en = Translator(to_lang='en')
# read class names from text file
classes = None
if language == 'ru':
    with open('yolov3-ru.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    with open('yolov3.txt', 'r') as f:
        classes_en = [line.strip() for line in f.readlines()]
else:
    with open('yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    with open('yolov3.txt', 'r') as f:
        classes_en = [line.strip() for line in f.readlines()]

dict_classes = dict()
for i in classes:
    dict_classes.update({i: True})

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes_en):
    label = str(classes_en[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def preprocess(frame, single, net, classes, classes_en, dict_classes):
    d = dict()
    height, width = frame.shape[0], frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    outs = net.forward(get_output_layers(net))
    classes_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if not dict_classes[classes[class_id]]:
                continue
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                s = h * w
                boxes.append([x, y, w, h, class_id, float(confidence), s])
                classes_ids.append(classes[class_id])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if single:
        if len(boxes) !=0:
            boxes = sorted(boxes, key=lambda x: x[-1])
            boxes = [boxes[-1]]
            a = 0
            for i in range(len(boxes)):
                if a == len(boxes):
                    break
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                a += 1
                draw_bounding_box(frame, box[4], box[5], round(x), round(y), round(x + w), round(y + h), classes_en)
                d = {classes[box[4]]: ' '}
        else:
            d = {'Nothing': ' '}
    else:
        a = 0
        if len(boxes) != 0:
            for i in indices:
                if a == len(boxes):
                    break
                i = i[0]
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                a += 1
                draw_bounding_box(frame, box[4], box[5], round(x), round(y), round(x + w), round(y + h), classes_en)
                s = set(classes_ids)
                d = dict()
                for i in s:
                    d.update({i: classes_ids.count(i)})
        else:
            d = {'Nothing': ' '}
    return frame, indices, boxes, classes_ids, d


def finder(frame,  find_class, language, classes, classes_en, dict_classes, Stop=False):
    height, width = frame.shape[0], frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    outs = net.forward(get_output_layers(net))
    classes_ids = []
    x = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if classes[class_id] == find_class:
                Stop = True
            else:
                continue
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                s = h * w
                boxes.append([x, y, w, h, classes[class_id], s])
                classes_ids.append(classes[class_id])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_bounding_box(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w),
                              round(y + h), classes_en)
    if x:
        if x <= 100:
            side = 'on your right'
        elif x > 100 and x < 250:
            side = 'on your left'
        else:
            side = 'in front of you'
        translation = translator.translate(find_class)
        say_bitch(translator.translate('the nearest {} is {}'.format(translation, side)), language)
    return frame, Stop, x


class MyTextInput(TextInput):
    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        if self.suggestion_text and keycode[1] == 'tab':
            self.insert_text(self.suggestion_text + ' ')
            return True
        return super(MyTextInput, self).keyboard_on_key_down(window, keycode, text, modifiers)


class MySettingsWithTabbedPanel(SettingsWithTabbedPanel):
    """
    It is not usually necessary to create subclass of a settings panel. There
    are many built-in types that you can use out of the box
    (SettingsWithSidebar, SettingsWithSpinner etc.).
    You would only want to create a Settings subclass like this if you want to
    change the behavior or appearance of an existing Settings class.
    """
    def on_close(self):
        Logger.info("main.py: MySettingsWithTabbedPanel.on_close")

    def on_config_change(self, config, section, key, value):
        Logger.info(
            "main.py: MySettingsWithTabbedPanel.on_config_change: "
            "{0}, {1}, {2}, {3}".format(config, section, key, value))


class InterfaceManager(MDApp, BoxLayout, App):#, FocusBehavior, LayoutSelectionBehavior,RecycleBoxLayout):

    def __init__(self, *args, **kwargs):

        self.clear_widgets()
        self.Start = False
        self.icon = 'logo_transparent.ico'
        try:
            self.capture.release()
        except:
            pass
        #self.main_layout.clear_widgets()
        super().__init__(**kwargs)
        self.config = ConfigParser()
        self.main_layout = BoxLayout(orientation="vertical")
        self.config.read('my.ini')
        lang = self.config.get('My Label', 'text')
        self.lang = lang
        size_label = self.config.get('My Label', 'font_size')
        if lang == 'ru':
            with open('yolov3-ru.txt', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            with open('yolov3.txt', 'r') as f:
                self.classes_en = [line.strip() for line in f.readlines()]
        else:
            with open('yolov3.txt', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            with open('yolov3.txt', 'r') as f:
                self.classes_en = [line.strip() for line in f.readlines()]
        if size_label == 'huge':
            self.font_size = 40
        elif size_label == 'medium':
            self.font_size = 30
        else:
            self.font_size = 20
        self.dict_classes = dict()
        for i in self.classes:
            self.dict_classes.update({i: True})
        self.translator= Translator(to_lang=self.lang)
        super(InterfaceManager, self).__init__(**kwargs)
        self.btn = Button(text=self.translator.translate("Tap the screen\n to start"), font_size=self.font_size)
        self.btn.bind(on_press=self.stop)

        self.btn8 = Button(text=self.translator.translate('Settings'),
                           background_color=(0, 1, 1, 1),
                           color=(1, 1, 1, 1),
                           size_hint_y=None,
                           height=60,
                           font_size=self.font_size
                           )
        self.btn8.bind(on_press=self.settings)
        self.main_layout.add_widget(self.btn8)
        self.main_layout.add_widget(self.btn)
        self.add_widget(self.main_layout)
        self.theme_cls.primary_palette = "Gray"
        self.settings_cls = MySettingsWithTabbedPanel

    def settings(self, ev):
        self.clear_widgets()
        self.config = ConfigParser()
        self.config.read('my.ini')
        main_layout = BoxLayout(orientation="vertical")
        dropdown = DropDown()
        main_layout_1 = BoxLayout(orientation="horizontal")
        main_layout.add_widget(Label(text=self.translator.translate('Settings'), color=(0, 0, 0, 1), font_size=self.font_size*2))
        for index in ['en', 'ru']:
            btn = Button(text='{}'.format(index), size_hint_y=None, height=60, font_size=self.font_size)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)
        self.mainbutton = Button(text=self.config.get('My Label', 'text'), size_hint=(None, None), height=60, font_size=self.font_size, background_color=(0, 0, 1, 1))
        self.mainbutton.bind(on_release=dropdown.open)
        dropdown.bind(on_select=self.change_lg)
        main_layout_1.add_widget(Label(text=self.translator.translate('Language'), color=(0, 0, 0, 1), font_size=self.font_size))
        main_layout_1.add_widget(self.mainbutton)
        main_layout.add_widget(main_layout_1)
        dropdown1 = DropDown()
        main_layout_1 = BoxLayout(orientation="horizontal")
        for index in ['huge', 'medium', 'small']:
            btn = Button(text='{}'.format(index), size_hint_y=None, height=60, font_size=self.font_size)
            btn.bind(on_release=lambda btn: dropdown1.select(btn.text))
            dropdown1.add_widget(btn)
        self.mainbutton1 = Button(text=self.config.get('My Label', 'font_size'), size_hint=(None, None), height=60, font_size=self.font_size, background_color=(0, 0, 1, 1))
        self.mainbutton1.bind(on_release=dropdown1.open)
        dropdown1.bind(on_select=self.change_sz)
        main_layout_1.add_widget(Label(text=self.translator.translate('Font Size'), color=(0, 0, 0.0, 1), font_size=self.font_size))
        main_layout_1.add_widget(self.mainbutton1)
        main_layout.add_widget(main_layout_1)
        self.btn5 = Button(text=self.translator.translate('Main'),
                           background_color=(0, 1, 1, 1),
                           color=(1, 1, 1, 1),
                           size_hint_y=None,
                           height=40,
                           font_size=self.font_size
                           )
        self.btn5.bind(on_press=self.__init__)
        main_layout.add_widget(self.btn5)
        self.add_widget(main_layout)

    def change_lg(self, ev, instance, *args):
        self.config = ConfigParser()
        self.config.read('my.ini')
        button_text = instance
        setattr(self.mainbutton, 'text', button_text)
        self.config.set('My Label', 'text', button_text)
        self.config.write()

    def change_sz(self, ev, instance, *args):
        self.config = ConfigParser()
        self.config.read('my.ini')
        button_text = instance
        setattr(self.mainbutton1, 'text', button_text)
        self.config.set('My Label', 'font_size', button_text)
        self.config.write()

    def start(self, event):
        try:
            self.capture.release()
        except:
            pass
        self.main_layout = BoxLayout(orientation="vertical")
        self.counter = 0
        self.prev = ''
        self.net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
        self.clear_widgets()
        self.title = "KivyMD Examples - Progress Loader"


        self.Start = True
        try:
            if self.find != '':
                self.find = self.find
            else:
                self.find = ''
        except:
            self.find = ''
        self.text_input = TextInput(multiline=False, size_hint=(1, 0.1*(self.font_size/20)),font_size=self.font_size, pos_hint={'center_x': 0.5, 'center_y': 0.95})
        self.text_input.bind(text=self.on_text)
        self.text_input_1 = TextInput(multiline=False, size_hint=(1, 0.1*(self.font_size/20)),font_size=self.font_size, pos_hint={'center_x': 0.5, 'center_y': 0.95})
        self.text_input_1.bind(text=self.on_text)
        self.text_input_2 = TextInput(multiline=False, size_hint=(1, 0.1*(self.font_size/20)),font_size=self.font_size, pos_hint={'center_x': 0.5, 'center_y': 0.95})
        self.text_input_2.bind(text=self.on_text)
        self.text_input_4 = TextInput(multiline=False, size_hint=(1, 0.1*(self.font_size/20)),font_size=self.font_size, pos_hint={'center_x': 0.5, 'center_y': 0.95})
        self.text_input_4.bind(text=self.on_text)
        self.second = Button(text="Second")
        self.final = Label(text="Hello World")
        self.single = False
        self.lab = 'Показывать один объект'
        self.img1 = Image(size_hint = (2, 2), pos_hint={'center_x': 0.5, 'center_y': 0.8})

        self.btn = Button(text=self.translator.translate("Aloud"),
                         font_size=self.font_size,
                         background_color=(1, 1, 1, 1),
                         color=(1, 1, 1, 1),
                         size=(32, 32),
                         size_hint=(.2, .5),
                         pos=(300, 300))

        self.btn.bind(on_press=self.callback_1)

        self.btn_2 = ToggleButton(text=self.translator.translate('One'),
                                 font_size=self.font_size,
                                 background_color=(1, 1, 1, 1),
                                 color=(1, 1, 1, 1),
                                 size=(32, 32),
                                 size_hint=(.2, .5),
                                 pos=(300, 250))
        self.btn_3 = ToggleButton(text=self.translator.translate('Fast'),
                                 font_size=self.font_size,
                                 background_color=(1, 1, 1, 1),
                                 color=(1, 1, 1, 1),
                                 size=(32, 32),
                                 size_hint=(.2, .5),
                                 pos=(300, 250))
        self.fast = False
        self.btn_2.bind(on_press=self.callback_2)

        self.btn_3.bind(on_press=self.callback_3)
        self.first = Button(text=self.translator.translate("Look for"),
                           font_size=self.font_size,
                           background_color=(1, 1, 1, 1),
                           color=(1, 1, 1, 1),
                           size=(32, 32),
                           size_hint=(.2, .5),
                           pos=(20, 25))

        if self.find:
            self.btn5 = Button(text=self.translator.translate('Look for {}'.format(self.find)),
                                background_color=(0, 1, 1, 1),
                               font_size=self.font_size,
                                color=(1, 1, 1, 1),
                                size_hint_y=None,
                                height=40
                                )
        else:
            self.btn5 = Button(text=self.translator.translate('Exit'),
                                background_color=(0, 1, 1, 1),
                                color=(1, 1, 1, 1),
                               font_size=self.font_size,
                                size_hint_y=None,
                                height=40
                                )
        self.btn5.bind(on_press=self.start)
        if self.find:
            self.btn_5 = Button(text=self.translator.translate('Look for {}'.format(self.find)),
                                background_color=(0, 1, 1, 1),
                                color=(1, 1, 1, 1),
                                font_size=self.font_size,
                                size_hint_y=None,
                                height=40
                                )
        else:
            self.btn_5 = Button(text=self.translator.translate('Exit'),
                                background_color=(0, 1, 1, 1),
                                color=(1, 1, 1, 1),
                                font_size=self.font_size,
                                size_hint_y=None,
                                height=40
                                )
        self.btn_5.bind(on_press=self.start)
        self.first.bind(on_press=self.show_second)
        self.capture = cv2.VideoCapture(0)

        self.bind(on_load=self.cb_loaded)
        self.matches = []
        self.main_layout = BoxLayout(orientation="vertical")
        self.main_layout_1 = BoxLayout(orientation="horizontal")
        if not self.find:
            self.clear_widgets()
            self.main_layout.clear_widgets()
            self.main_layout.add_widget(self.img1)
            self.main_layout_1.add_widget(self.btn)
            self.main_layout_1.add_widget(self.btn_2)
            self.main_layout.add_widget(self.main_layout_1)
            self.main_layout_1 = BoxLayout(orientation="horizontal")
            self.main_layout_1.add_widget(self.first)
            self.main_layout_1.add_widget(self.btn_3)
            self.main_layout.add_widget(self.main_layout_1)
            self.add_widget(self.main_layout)
            self.btn8 = Button(text=self.translator.translate('Main'),
                               background_color=(0, 1, 1, 1),
                               color=(1, 1, 1, 1),
                               size_hint_y=None,
                               height=40,
                               font_size=self.font_size
                               )
            self.btn8.bind(on_press=self.__init__)
            self.main_layout.add_widget(self.btn8)
            Clock.schedule_interval(self.update, 1.0 / 33.0)
        else:
            if self.Start:
                self.clear_widgets()
                self.main_layout.clear_widgets()
                self.Stop = False
                self.p = 0
                self.btn_6 = Button(text=self.translator.translate('Finish'),
                                    background_color=(1, 1, 1, 1),
                                    color=(1, 1, 1, 1),
                                    font_size=self.font_size,
                                    size_hint_y=None,
                                    height=40
                                    )
                self.btn_6.bind(on_press=self.stop)
                self.net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
                self.clear_widgets()
                self.main_layout.add_widget(self.img1)
                self.main_layout.add_widget(self.btn_6)
                self.add_widget(self.main_layout)
                Clock.schedule_interval(self.update, 1.0 / 33.0)

    def stop(self, event):
        self.find = ''
        self.start(0)

    def cb_loaded(self):
       self.clear_widgets()
       self.add_widget(self.btn_3)

    def update(self, df, pryams=[]):
        if self.Start:
            ret, frame = self.capture.read()
            frame = cv2.resize(frame, (800, 450))
            if not self.find:
                frame, self.indices, self.boxes, self.classes_ids, self.d = preprocess(frame, self.single, self.net, self.classes, self.classes_en, self.dict_classes)
            else:
                if self.Start:
                    frame, self.Stop, self.p = finder(frame, self.find, self.lang, self.classes, self.classes_en, self.dict_classes)

            frame = cv2.flip(frame, 0)
            buf = frame.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = texture1

    def callback_1(self, event):
        self.update(0, pryams=[[100, 100, 200, 200]])
        d = self.d
        for i in d:
            if d[i] == 1:
                translation = self.translator.translate(str(d[i]) + ' ' + i)
            else:
                if i[-1] in ['x', 's', 'o', 'ss', 'sh', 'ch']:
                    translation = self.translator.translate(str(d[i]) + ' ' + i + 'es')
                else:
                    translation = self.translator.translate(str(d[i]) + ' ' + i)
            say_bitch(self.translator.translate('{}'.format(translation)), self.lang)

    def callback_2(self, event):
        if self.single:
            self.single = False
            self.lab = 'Показывать несколько объектов'
        else:
            self.single = True
            self.lab = 'Показывать один объект'

    def callback_3(self, event):
        if self.fast:
            self.net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
            self.fast = False
            self.clear_widgets()
            self.main_layout.clear_widgets()
            self.main_layout_1.clear_widgets()
            self.btn_3 = ToggleButton(text=self.translator.translate('Fast'),
                                     font_size=self.font_size,
                                     background_color=(1, 1, 1, 1),
                                     color=(1, 1, 1, 1),
                                     size=(32, 32),
                                     size_hint=(.2, .5),
                                     pos=(300, 250))
            self.btn_3.bind(on_press=self.callback_3)
            self.add_widget(self.btn_3)
            self.clear_widgets()
            self.main_layout.clear_widgets()
            self.main_layout.add_widget(self.img1)
            self.main_layout_1.add_widget(self.btn)
            self.main_layout_1.add_widget(self.btn_2)
            self.main_layout_1.add_widget(self.first)
            self.main_layout_1.add_widget(self.btn_3)
            self.main_layout.add_widget(self.main_layout_1)
            self.btn8 = Button(text=self.translator.translate('Main'),
                               background_color=(0, 1, 1, 1),
                               color=(1, 1, 1, 1),
                               size_hint_y=None,
                               height=40,
                               font_size=self.font_size
                               )
            self.btn8.bind(on_press=self.__init__)
            self.main_layout.add_widget(self.btn8)
            self.add_widget(self.main_layout)
        else:
            self.fast = True
            self.net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
            self.clear_widgets()
            self.main_layout.clear_widgets()
            self.main_layout_1.clear_widgets()
            self.btn_3 = ToggleButton(text=self.translator.translate('Accurate'),
                                     font_size=self.font_size,
                                     background_color=(1, 1, 1, 1),
                                     color=(1, 1, 1, 1),
                                     size=(32, 32),
                                     size_hint=(.2, .5),
                                     pos=(300, 250))
            self.btn_3.bind(on_press=self.callback_3)
            self.clear_widgets()
            self.main_layout.clear_widgets()
            self.main_layout.add_widget(self.img1)
            self.main_layout_1.add_widget(self.btn)
            self.main_layout_1.add_widget(self.btn_2)
            self.main_layout_1.add_widget(self.first)
            self.main_layout_1.add_widget(self.btn_3)
            self.main_layout.add_widget(self.main_layout_1)
            self.btn8 = Button(text=self.translator.translate('Main'),
                               background_color=(0, 1, 1, 1),
                               color=(1, 1, 1, 1),
                               size_hint_y=None,
                               height=40,
                               font_size=self.font_size
                               )
            self.btn8.bind(on_press=self.__init__)
            self.main_layout.add_widget(self.btn8)
            self.add_widget(self.main_layout)

    def show_second(self, button):
        try:
            self.capture.release()
        except:
            pass
        self.Start = False
        self.clear_widgets()
        self.main_layout.clear_widgets()
        self.word_list = self.classes
        main_layout = BoxLayout(orientation="vertical")
        self.main_layout.add_widget(self.text_input)
        self.btn5.bind(on_press=self.start)
        self.main_layout.add_widget(self.btn5)
        self.add_widget(self.main_layout)

    def on_text(self, instance, value):
        if value!=self.prev and value!='' :
            self.suggestion_text = ''
            val = value[value.rfind(' ') + 1:]
            self.prev = value
            if value != '':
                matches = [self.word_list[i] for i in range(len(self.word_list)) if
                           value in self.word_list[i][:len(value)]]
            else:
                matches = []
            self.matches = matches
            #self.text_input = TextInput(multiline=False, size_hint=(1, 0.1), pos_hint={'center_x': 0.5, 'center_y': 0.95})
            #self.text_input.bind(text=self.on_text)
            #main_layout_1 = BoxLayout(orientation="vertical")
            if len(matches) != 0:
                self.clear_widgets()
                self.main_layout.clear_widgets()
                main_layout_1 = BoxLayout(orientation="vertical")
                self.text_input_1 = TextInput(multiline=False, size_hint=(1, 0.1*(self.font_size/20)),
                                              font_size=self.font_size,
                                              pos_hint={'center_x': 0.5, 'center_y': 0.95})
                self.text_input_1.bind(text=self.on_text)
                self.main_layout.add_widget(self.text_input)
                if self.find:
                    self.btn_5 = Button(text=self.translator.translate('Look for {}'.format(self.find)),
                                       background_color=(0, 1, 1, 1),
                                        font_size=self.font_size,
                                       color=(1, 1, 1, 1),
                                       size_hint_y=None,
                                       height=40
                                       )
                else:
                    self.btn_5 = Button(text=self.translator.translate('Exit'),
                                        background_color=(0, 1, 1, 1),
                                        color=(1, 1, 1, 1),
                                        font_size=self.font_size,
                                        size_hint_y=None,
                                        height=40
                                        )
                self.btn_5.bind(on_press=self.start)

                main_layout = BoxLayout(orientation="vertical")
                for i in matches:
                    self.btn = Button(text=i,
                                      background_color=(1, 1, 1, 1),
                                      color=(1, 1, 1, 1),
                                      font_size=self.font_size,
                                      size_hint_y=None,
                                      height=40
                                      )

                    # bind() use to bind the button to function callback
                    self.btn.bind(on_press=self.Pressbtn)
                    main_layout.add_widget(self.btn)
                self.main_layout.add_widget(main_layout)


                self.main_layout.add_widget(self.btn_5)
                self.add_widget(self.main_layout)
                #self.add_widget(main_layout_1)
                return
            else:
                return
            if not val:
                return
            try:
                word = [word for word in self.word_list
                        if word.startswith(val)][0][len(val):]
                if not word:
                    return
                self.suggestion_text = word
            except IndexError:
                print('Index Error.')

    def Pressbtn(self, instance):
        button_text = instance.text
        self.main_layout.clear_widgets()
        self.find = button_text
        if self.find:
            self.btn5 = Button(text=self.translator.translate('Look for {}'.format(self.find)),
                                background_color=(0, 1, 1, 1),
                                color=(1, 1, 1, 1),
                                size_hint_y=None,
                               font_size=self.font_size,
                                height=40
                                )
        else:
            self.btn5 = Button(text=self.translator.translate('Exit'),
                                background_color=(0, 1, 1, 1),
                                color=(1, 1, 1, 1),
                               font_size=self.font_size,
                                size_hint_y=None,
                                height=40
                                )
        self.clear_widgets()
        self.btn5.bind(on_press=self.start)
        # TODO: добавить действие для btn5
        main_layout_1 = BoxLayout(orientation="vertical")
        self.text_input.bind(text=self.on_text)
        self.main_layout.add_widget(self.text_input)
        main_layout = BoxLayout(orientation="vertical")
        for i in self.matches:
            self.btn = Button(text=i,
                              background_color=(1, 1, 1, 1),
                              color=(1, 1, 1, 1),
                              font_size=self.font_size,
                              size_hint_y=None,
                              height=40
                              )

            # bind() use to bind the button to function callback
            self.btn.bind(on_press=self.Pressbtn)
            main_layout.add_widget(self.btn)
        self.main_layout.add_widget(main_layout)
        self.main_layout.add_widget(self.btn5)
        self.add_widget(self.main_layout)

    def on_stop(self):
        # without this, app will not exit even if the window is closed
        self.capture.release()



class MyApp(App):
    def build(self):
        return InterfaceManager(orientation='horizontal')


if __name__ == '__main__':
    MyApp().run()
