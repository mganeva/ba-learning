import click
from training.experiment import run_experiment

@click.command()
@click.argument("experiment_config", required=True, type=str)
def main(experiment_config):
    run_experiment(experiment_config)
    
if __name__ == '__main__':
    main()

