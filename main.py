import sys
import extract as ex
import eval as ev

def nand(a,b):
    return not (a and b)

train_file,annotation_file,output_file = sys.argv[1:]

ex.main(train_file,output_file)
ev.main(annotation_file,output_file)

a = True
b = False

print nand(a,a) != a, "negation"
print nand(b,b) != b, "negation b"
print nand(nand(a,a),nand(a,a)) == a, "identity"
print nand(a,b) != a and b
print (not (nand(a,b) and nand(a,b))) == a and b
print nand(nand(a,b),nand(a,b)) == a and b, "and"
print nand(nand(a,a),nand(b,b)) == a or b, "or"


