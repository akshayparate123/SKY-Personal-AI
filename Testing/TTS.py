import multiprocessing
import pyttsx3
import time
from threading import Thread


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

def speak(phrase):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(phrase)
    engine.runAndWait()
    engine.stop()

def stop_speaker():
    global term
    term = True
    t.join()

@threaded
def manage_process(p):
	global term
	while p.is_alive():
		if term:
			p.terminate()
			term = False
		else:
			continue


def say(phrase):
	global t
	global term
	term = False
	p = multiprocessing.Process(target=speak, args=(phrase,))
	p.start()
	t = manage_process(p)
		
# if __name__ == "__main__":
# 	start_time = time.time()
# 	say("Artificial Intelligence (AI) is a branch of computer science ")
# 	print("--- %s seconds ---" % (time.time() - start_time))
 # time.sleep(2)
	# stop_speaker()
	# say("this process is running right now")
	# time.sleep(1.5)
	# stop_speaker()
 
 