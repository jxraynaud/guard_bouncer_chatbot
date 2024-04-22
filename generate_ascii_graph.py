from workflows.main.main_workflow import MainWorkflow


def main():
    workflow = MainWorkflow()
    print(workflow.generate_ascii())

if __name__ == '__main__':
    main()
