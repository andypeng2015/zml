import jax
import jax.numpy as jnp

# jax.print_environment_info()

dtype = 'bfloat16'
# float32 or bfloat16

if __name__ == '__main__':
    key =jax. random. PRNGKey(0)
    input = jax.random.uniform(key,(256,256),dtype=getattr(jnp, dtype))
    
    @jax.named_scope("toto")
    def exp_func_coucou(x):
        return jax.numpy.exp(x)

    print("=jitted=")
    jitted = jax.jit(exp_func_coucou)
    print(jitted)

    print("=lowered=")
    lowered = jitted.lower(input)
    print(lowered.as_text())

    print("= compiled =")
    compiled = lowered. compile()
    print(compiled.as_text())

    output = compiled(input)
    print(output)
