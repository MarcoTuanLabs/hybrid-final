import random

KEYS=['cat','dog','fox','bear','wolf','lion','tiger','zebra','horse','mouse','sheep','goat','duck','crow','owl','deer']

VALUES=list('ABCDEFGH')

def make_example(seed,surface,control=False):
    rng=random.Random(seed)
    keys=rng.sample(KEYS,8);values=rng.sample(VALUES,8)
    target,foil,other=rng.sample(range(8),3)
    swapped=values.copy()
    i,j=(foil,other) if control else (target,foil)
    swapped[i],swapped[j]=swapped[j],swapped[i]
    # Under the nuisance control the queried binding is unchanged. Its contrast
    # still uses the target value and the original foil value for comparability.
    va,vb=values[target],values[foil]
    def prefix(v):
        if surface=='table':
            data=''.join(f'{k}: {x}\n' for k,x in zip(keys,v))
        elif surface=='prose':
            data=' '.join(f'The label assigned to the {k} is {x}.' for k,x in zip(keys,v))+'\n'
        else:raise ValueError(surface)
        return 'Use the following assignments. Answer with exactly one uppercase letter.\n\n'+data+'\nThe assignments are complete. The next question requests one of these letters.\n'
    suffix=f'Question: What is the label of {keys[target]}?\nAnswer:'
    return {'prefix_a':prefix(values),'prefix_b':prefix(swapped),'suffix':suffix,
            'value_a':va,'value_b':vb,'target':keys[target],'foil':keys[foil],
            'native_gold_a':values[target],'native_gold_b':swapped[target],
            'surface':surface,'control':control,'seed':seed,'keys':keys,
            'assignments_a':values,'assignments_b':swapped}
