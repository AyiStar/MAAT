import logging
from .utils.data import AdapTestDataset
from tensorboardX import SummaryWriter


class AdapTestDriver(object):

    @staticmethod
    def run(model, strategy, adaptest_data,
            test_length, log_dir):
        writer = SummaryWriter(log_dir)

        logging.info(f'start adaptive testing with {strategy.name} strategy')

        logging.info(f'Iteration 0')
        # evaluate models
        results = model.adaptest_evaluate(adaptest_data)
        # log results
        for name, value in results.items():
            logging.info(f'{name}:{value}')
            writer.add_scalars(name, {strategy.name: value}, 0)

        for it in range(1, test_length + 1):
            logging.info(f'Iteration {it}')
            # select question
            selected_questions = strategy.adaptest_select(model, adaptest_data)
            for student, question in selected_questions.items():
                adaptest_data.apply_selection(student, question)
            # update models
            model.adaptest_update(adaptest_data)
            # evaluate models
            results = model.adaptest_evaluate(adaptest_data)
            # log results
            for name, value in results.items():
                logging.info(f'{name}:{value}')
                writer.add_scalars(name, {strategy.name: value}, it)
