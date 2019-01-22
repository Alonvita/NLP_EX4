import sys
import extract as ex
import eval as ev


train_file,annotation_file,output_file = sys.argv[1:]

ex.main(train_file,output_file)
ev.main(annotation_file,output_file)
