# coding: utf-8

import click
import os
from .cli import pass_context

from minimal.server.server import run_server


@click.command('decision_tree', short_help='Decision Tree Classifier')
@click.option('-m', '--mode',
              default='train',
              type=click.Choice(['train', 'test']),
              help='train or test your decision tree model')
@click.option('-f', '--file',
              type=str,
              help='file path of your data set')
@pass_context
def cli(ctx, mode, port):
    print 'hello %s', % mode

@click.command('knn', short_help='K Nearest Neighbors Classifier')
@click.option('-m', '--mode',
              default='train',
              type=click.Choice(['train', 'test']),
              help='train or test your knn model')
@click.option('-f', '--file',
              type=str,
              help='file path of your data set')
@pass_context
def cli(ctx, mode, port):
    print 'hello %s', % mode
