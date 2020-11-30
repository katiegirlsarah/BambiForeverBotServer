import requests
import time

def getMp3(line):

    n = 2500

    msgs = [line[i:i+n] for i in range(0, len(line), n)]


    for i in range(len(msgs)):
        a = requests.post("https://ttsmp3.com/makemp3_new.php",{"msg":msgs[i],"lang":"Salli","source":"ttsmp3"})
        print(a.text)
        z = a.text.split("URL\":\"")[1].split("\",\"")[0].replace("\\","").replace("\\","").replace("\\","").replace("\\","")

        b = open("out"+str(i)+".mp3","wb").write(requests.get(z).content)
        a.close()
        
        time.sleep(20)
