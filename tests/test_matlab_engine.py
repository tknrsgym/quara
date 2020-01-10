import matlab.engine

eng = matlab.engine.start_matlab()
eng.main_qpt_1qubit(nargout=0)
eng.quit()
