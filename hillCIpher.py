import numpy as np ;
import math ;

f= open("plainText.txt","r");
p = f.read() ;
original_length = len(p);
ascii_dict = [chr(i) for i in range(256)]


def mineur(matrice, i, j):
  sous_matrice = matrice.copy()
  sous_matrice = np.delete(sous_matrice, i, axis=0)
  sous_matrice = np.delete(sous_matrice, j, axis=1)
  return np.linalg.det(sous_matrice)

def comatrice(matrice):
  row,column = matrice.shape
  comatrice = np.zeros((row, column))
  for i in range(row):
    for j in range(column):
      signe = (-1)**(i + j)
      comatrice[i, j] = signe * mineur(matrice, j, i)
  return comatrice




def mod(t):
    t = int(t % 256) ;
    return t

def check_key(k):
    num_rows , num_columns = np.shape(k);
    if num_columns != num_rows :
        return False ;
    det = mod(round(np.linalg.det(k)));
    pgcd = math.gcd(det,256);
    key_valid = (det!=0) and (pgcd == 1) ;
    return key_valid ;

def text_to_number(p):
   a =[];
   if len(p) % 2 != 0 :
       p = p + "Z" ;
   for t in p : 
        a.append( mod(ascii_dict.index(t)) );

   return np.array(a).reshape(-1,2).T;


def number_to_text(c):
    c = np.ravel(c.T);
    r ="" ;
    for t in c :
         r += ascii_dict[mod(t)];
    
    if r[-1] == 'Z' and len(r) > original_length:
        r = r[:-1]
    return r ;





def encrypt(p,k) :
    if check_key(k) :
       p = text_to_number(p);
       c = np.matmul(k,p) % 256 ;
       #print("c: ",c);
       r = number_to_text(c);

       return r ;
    else :
        print("Invalid key !");
        
        



def multiplicative_inverse(a, n):
    t = 0;
    r = n ;
    new_t = 1 ;
    new_r = a ;
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r

    if r > 1:
        raise Exception("Error !") ;

    return mod(t);


def decryption(c,k) :
    if check_key(k):

       c = text_to_number(c);
       #print("c' : ",c)
       det = int(np.round(np.linalg.det(k))) % 256 ;
       #print("det :",det)
       det_inv = multiplicative_inverse(det, 256) ;
       #print("det inv : ",det_inv)
       adj_key_matrix = np.round(det_inv * comatrice(k)).astype(int) % 256 ;
       p = np.matmul(adj_key_matrix,c) % 256 ;
       #print("p : \n",p);
       p = number_to_text(p);
       return p ;
    else :
        raise Exception("Invalid key !")


k = [9,4,5,7] ;
k = np.array(k) ;

if (len(k) % 2) != 0 :
    raise Exception("Invalid key !"); 

k = k.reshape(-1, int(len(k) / 2) ).T ;
print("plaintext :",p)

c = encrypt(p,k);
print("Encrypted :",c);

p = decryption(c,k);
print("Decryipted :",p);




import numpy as np ;
import math ;

f= open("plainText.txt","r");
p = f.read() ;
original_length = len(p);





def mod(t):
    t = int(t % 256) ;
    return t

def check_key(k):
    num_rows , num_columns = np.shape(k);
    if num_columns != num_rows :
        return False ;
    det = mod(np.linalg.det(k));
    pgcd = math.gcd(det,256);
    key_valid = (det!=0) and (pgcd == 1) ;
    return key_valid ;

def text_to_number(p):
   a =[];
   if len(p) % 2 != 0 :
       p = p + "Z" ;
   for t in p : 
        a.append( mod(ord(t)) );

   return np.array(a).reshape(-1,2).T;


def number_to_text(c):
    c = np.ravel(c.T);
    if len(c) != original_length :
        c =c[:-1]
    r ="" ;
    for t in c :
         r += chr(mod(t));
    return r ;



a = text_to_number(p);


def encrypt(p,k) :
    if check_key(k) :
       p = text_to_number(p);
       c = np.matmul(k,p) % 256 ;
       #print("c: ",c);
       r = number_to_text(c);

       return r ;
    else :
        print("Invalid key !");
        
        



def multiplicative_inverse(a, n):
    t = 0;
    r = n ;
    new_t = 1 ;
    new_r = a ;
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r

    if r > 1:
        raise Exception("Error !") ;

    return mod(t);


def decryption(c,k) :
    if check_key(k):

       c = text_to_number(c);
       #print("c' : ",c)
       det = int(np.round(np.linalg.det(k))) % 256 ;
       print("det :",det)
       det_inv = multiplicative_inverse(det, 256)
       print("det inv : ",det_inv)
       adj_key_matrix = np.round(det_inv * np.linalg.inv(k)).astype(int) % 256 ;
       p = np.matmul(adj_key_matrix,c) % 256 ;
       print("p : \n",p);
       p = number_to_text(p);
       return p ;
    else :
        raise Exception("Invalid key !")



k = np.array([9,4,5,7]) ;

if (len(k) % 2) != 0 :
    raise Exception("Invalid key !"); 

k = k.reshape(-1, int(len(k) / 2) ).T ;
print("plaintext :",p)
c = encrypt(p,k);
print("Encrypted : ",c);

p = decryption(c,k);
print("Decryipted : ",p);