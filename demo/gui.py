# import the library
from appJar import gui
import time
# this two lines are for a mysterious error
# NEVER MOVING THIS TWO LINES!!!!!
import matplotlib
matplotlib.use('TkAgg')

import demo.ACBC_algorithm as M
import random
from random import randint


def generateUser():
    # randomly delay sometime
    userTimeSlot = randint(1000, 10000)
    Message = '[Time slot between users: '+str(userTimeSlot)+' ms]'
    app.addListItem('Message', Message)
    app.setPollTime(userTimeSlot)

    # randomly choose a file from file list
    Rf = random.choice(list(M.file_dictionary.keys()))
    Sf = M.file_dictionary[Rf]
    #total_size += Sf
    A = M.Scenario(trace_length=1000,
                   size_of_file=Sf,
                   mean=28,
                   var=3.8)
    B = M.MAP(total_chunk=A.chunk_number,
              expectation_Xi=A.chunk_mean,
              standard_deviation_Xi=A.chunk_std,
              total_EN=20,
              threshold=0.9,
              EN_start=0,
              chunk_start=0)
    _, b, _, _ = M.run(A, B)
    for item in b.items():
        list_id = 'listEN' + str(item[0])
        if len(item[1]) > 0:
            for chunk in item[1]:
                cache_chunk = str(Rf) + '-' + str(chunk)
                # duplicate chunk detection
                if cache_chunk in app.getAllListItems(list_id):
                    continue
                app.addListItem(list_id, cache_chunk)
    Message = '<Incoming User>'\
              + ' File-ID=' \
              + str(Rf) \
              + ' Size-of-File=' \
              + str(Sf) \
              + ' [succeed]'
    app.addListItem('Message', Message)


def pressStart(btn):
    app.registerEvent(generateUser)
    app.setButtonState('Start', 'disabled')


def pressClear(btn):
    app.clearListBox('Message')
    for i in range(20):
        label_id = 'listEN' + str(i)
        app.clearListBox(label_id)


def pressClose(btn):
    app.stop()


def pressInfo(btn):
    app.showSubWindow('Info')
app = gui('Demo', '875x700')

app.setSticky('news')
app.setExpand('both')
app.setFont(12)

app.startLabelFrame('Edge Nodes')
label_id = ''
label = ''
for i in range(10):
    label_id = 'lEN' + str(i)
    label = 'EN ' + str(i)
    app.addLabel(label_id, label, 0, i)
    app.setLabelWidth(label_id, 10)
    label_id = 'listEN' + str(i)
    app.addListBox(label_id, [''], 1, i)
    app.setListBoxWidth(label_id, 10)
for i in range(10, 20):
    label_id = 'lEN' + str(i)
    label = 'EN ' + str(i)
    app.addLabel(label_id, label, 4, i - 10)
    app.setLabelWidth(label_id, 10)
    label_id = 'listEN' + str(i)
    app.addListBox(label_id, [''], 5, i - 10)
    app.setListBoxWidth(label_id, 10)
app.stopLabelFrame()


app.startLabelFrame('User Infomation')
app.addLabel('lMessage', 'Message', 8, 0, 1, 1)
app.addListBox('Message',[], 9, 0)
app.setListBoxWidth('Message', 140)
app.setListBoxHeight('Message', 10)
app.stopLabelFrame()


app.startLabelFrame('Control Panel')
app.addButton('Start', pressStart, 10, 0)
app.setButtonWidth('Start', 15)
app.addButton('Clear', pressClear, 10, 1)
app.setButtonWidth('Clear', 15)
app.addButton('Close', pressClose, 10, 2)
app.setButtonWidth('Close', 15)
app.addButton('Info', pressInfo, 10, 3)
app.setButtonWidth('Info', 15)
app.stopLabelFrame()
app.startSubWindow('Info', modal=True)
app.setGeometry("150x170")

app.startLabelFrame('File List')
app.addLabel('sl1',
             'file ID    size\n  1       10000Mbit\n  2       12000Mbit\n  3       15000Mbit\n  4       18000Mbit\n  5       20000Mbit\n  6       24000Mbit',
             0, 0)
app.stopLabelFrame()

app.addLabel('sl2', '    EN number       : 20   ', 1, 0)
app.addLabel('sl3', 'Random slot range: 1-10s', 2, 0)

app.go()
