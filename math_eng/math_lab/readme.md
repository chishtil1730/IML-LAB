# R studio codes:

### Basic operations:

```bash
    #remainder 
    a%%b
    
    #Division 
    a%/%b
```
### Arrays:

1.Basics of arrays 
```bash
    #Declaring an array:
    x = c("val1","val2","val3")
    
    #Creating multiple arrays and combining:
    x = c("Max","Verstappen")
    y = c("Charles","Leclerc")
    z = c(x,y)
```

2. Indexing and viewing:
```bash
    x = c("Max","Verstappen","Chishti")
    x[i] #i=1,sizeof(array)
```

### Sequences in R studio:
1. Basic sequence:
```bash
  #sequence from 1 to 10
    seq(1,10)
    
  #sequence separated by a value:
  seq(1,10,by=0.5)
```

2. Repetitions:
```bash
    #assuming an array:
    x = c("Max","Verstappen")
    #array to refer how many times to be repeated:
    x2 = c(3,1)
    y = rep( x , times = x2)

  #Output:
  Max Max Max Verstappen 
```

### Matrices:
1. Basics of Matrix:
```bash 
    X = matrix( c(1,2,3,4), nrow = 2, ncol =2 , byrow = T)
    
    #Output:
    [1,2]
    [3,4]
```
2. Changing column names:
```bash
  M = matrix( c(1,2,3,4),nrow=2,ncol=2,byrow=T)
  colnames(M) = c("col1","col2")
  rownames(M) = c("row1",row2)
```
